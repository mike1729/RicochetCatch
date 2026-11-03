"""Immutable pre-generation commitments without evaluating hidden episodes."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import platform
import secrets
import shutil
import subprocess

import numpy as np

from .run_astra import RunnerError, build_prompt, isolation_config, sha256_file, write_json

REQUIRED_FILES = (
    "GAME_DESIGN.md", "EXPERIMENT_PLAN.md", "configs/game.json",
    "src/ricochet_catch/env.py", "src/ricochet_catch/observations.py",
    "src/ricochet_catch/policy.py", "src/ricochet_catch/assignments.py",
    "src/ricochet_catch/baselines.py", "src/ricochet_catch/evaluate.py",
    "src/ricochet_catch/probes.py",
    "src/ricochet_catch/run_astra.py", "src/ricochet_catch/freeze.py",
    "schemas/tensor-assignments-v2.json", "prompts/neutral.txt", "requirements.txt", "pyproject.toml", "uv.lock",
    "ricochet_sim.py",
)


def _read_manifest(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "ricochet-freeze-v1":
        raise RunnerError("Unknown freeze manifest.")
    if not isinstance(manifest.get("files"), dict) or not isinstance(manifest.get("artifacts"), dict):
        raise RunnerError("Freeze manifest must record source and artifact hashes.")
    return manifest


def _contained_file(base: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise RunnerError("Freeze manifest contains an unsafe file path.")
    file = (base / name).resolve()
    if not file.is_relative_to(base):
        raise RunnerError("Freeze manifest contains an unsafe file path.")
    return file


def _verify_artifacts(manifest_path: Path, manifest: dict) -> None:
    """Check artifact commitments independently of the superseded source tree."""
    for name, digest in manifest["artifacts"].items():
        file = _contained_file(manifest_path.parent, name)
        if not file.is_file() or sha256_file(file) != digest:
            raise RunnerError(f"Frozen artifact changed or missing: {name}")


def _freeze_names(root: Path) -> set[str]:
    names = set(REQUIRED_FILES)
    for pattern in ("src/ricochet_catch/**/*.py", "schemas/*.json", "configs/*.json"):
        names.update(str(file.relative_to(root)) for file in root.glob(pattern) if file.is_file())
    # Fixtures can have any extension or live in hidden directories. Exclude only
    # generated caches and version-control metadata, never fixture suffixes.
    excluded = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis", ".git"}
    for file in (root / "tests").rglob("*"):
        relative = file.relative_to(root)
        if (file.is_file() and not excluded.intersection(relative.parts)
                and file.suffix not in {".pyc", ".pyo"} and file.name != ".DS_Store"):
            names.add(str(relative))
    return names


def _git_provenance(root: Path, names: set[str]) -> dict:
    """Distinguish the checkout's HEAD from a revision containing the frozen bytes."""
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=root, capture_output=True,
                              timeout=5, check=False)

    unavailable = {"head_revision": None, "git_dirty": None,
                   "frozen_files_match_revision": None, "uncommitted_frozen_files": []}
    try:
        head = git("rev-parse", "HEAD")
        if head.returncode:
            return unavailable
        revision = head.stdout.decode().strip()
        prefix = git("rev-parse", "--show-prefix")
        algorithm = git("rev-parse", "--show-object-format")
        tree = git("ls-tree", "-rz", "--full-tree", "HEAD")
        status = git("status", "--porcelain=v1", "--untracked-files=all")
        if any(result.returncode for result in (prefix, algorithm, tree, status)):
            return {**unavailable, "head_revision": revision}
        object_format = algorithm.stdout.decode().strip()
        if object_format not in {"sha1", "sha256"}:
            return {**unavailable, "head_revision": revision}
        committed = {}
        for entry in tree.stdout.split(b"\0"):
            if entry:
                metadata, path = entry.split(b"\t", 1)
                committed[os.fsdecode(path)] = metadata.split()[2].decode()
        root_prefix = os.fsdecode(prefix.stdout).strip()
        mismatches = []
        for name in sorted(names):
            raw = (root / name).read_bytes()
            blob = hashlib.new(object_format, f"blob {len(raw)}\0".encode() + raw).hexdigest()
            if committed.get(root_prefix + name) != blob:
                mismatches.append(name)
        return {"head_revision": revision, "git_dirty": bool(status.stdout),
                "frozen_files_match_revision": not mismatches,
                "uncommitted_frozen_files": mismatches}
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return unavailable


def verify_freeze(root: str | Path, manifest_path: str | Path) -> dict:
    root, manifest_path = Path(root).resolve(), Path(manifest_path).resolve()
    manifest = _read_manifest(manifest_path)
    for name, digest in manifest.get("files", {}).items():
        file = _contained_file(root, name)
        if not file.is_file() or sha256_file(file) != digest:
            raise RunnerError(f"Frozen file changed or missing: {name}")
    _verify_artifacts(manifest_path, manifest)
    if _freeze_names(root) - set(manifest["files"]):
        raise RunnerError("Freeze manifest omits required implementation files.")
    if hashlib.sha256(build_prompt(root).encode()).hexdigest() != manifest.get("prompt_sha256"):
        raise RunnerError("Current prompt differs from the frozen prompt.")
    return manifest


def freeze_project(root: str | Path, out_dir: str | Path, calibration_path: str | Path | None = None,
                   preflight_path: str | Path | None = None,
                   previous_freeze_path: str | Path | None = None) -> dict:
    root, dest = Path(root).resolve(), Path(out_dir).resolve()
    if dest.exists():
        raise RunnerError("Freeze directory already exists; preserve commitments and use a new version.")
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise RunnerError("Cannot freeze missing files: " + ", ".join(missing))
    previous_path = Path(previous_freeze_path).resolve() if previous_freeze_path else None
    previous = None
    if previous_path:
        previous = _read_manifest(previous_path)
        _verify_artifacts(previous_path, previous)
        if not {"development-seeds.json", "test-seeds.private.json"}.issubset(previous["artifacts"]):
            raise RunnerError("Previous freeze omits seed commitments.")
        if previous.get("astra_generation") != "not requested" or previous.get("hidden_test_evaluation") != "not run":
            raise RunnerError("Only an unevaluated, pre-generation freeze may be superseded.")
    calibration = json.loads(Path(calibration_path).read_text()) if calibration_path else {}
    if not calibration_path and previous and "calibration.json" in previous["artifacts"]:
        calibration = json.loads((previous_path.parent / "calibration.json").read_text())
    dev = list(calibration.get("seeds", range(256)))
    if len(dev) != 256 or len(set(dev)) != 256 or any(type(seed) is not int or seed < 0 for seed in dev):
        raise RunnerError("Development commitment must contain exactly 256 distinct nonnegative integer seeds.")
    test: list[int] = []
    if previous:
        previous_dev = json.loads((previous_path.parent / "development-seeds.json").read_text())
        hidden = json.loads((previous_path.parent / "test-seeds.private.json").read_text())
        if previous_dev.get("seeds") != dev:
            raise RunnerError("Re-freezing must preserve the committed development seeds.")
        test = hidden.get("seeds", [])
        if (not isinstance(test, list) or len(test) != 256
                or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in test)
                or len(set(test)) != 256 or set(test).intersection(set(dev) | set(range(256)))
                or hidden.get("episodes") != 256 or hidden.get("status") != "frozen, unevaluated"
                or hidden.get("random_action_seed0") != 20_000_019):
            raise RunnerError("Previous hidden seed commitment is invalid or no longer unevaluated.")
    else:
        excluded = set(dev) | set(range(256))
        while len(test) < 256:
            seed = secrets.randbits(63)
            if seed not in excluded:
                test.append(seed)
                excluded.add(seed)
    dest.mkdir(parents=True, mode=0o700)
    if calibration:
        write_json(dest / "calibration.json", calibration)
    if previous:
        for name in ("development-seeds.json", "test-seeds.private.json"):
            shutil.copyfile(previous_path.parent / name, dest / name)
    else:
        write_json(dest / "development-seeds.json", {"seeds": dev, "distribution": "public fixed development set"})
        write_json(dest / "test-seeds.private.json", {"seeds": test, "episodes": 256,
            "status": "frozen, unevaluated", "random_action_seed0": 20_000_019})
    (dest / "test-seeds.private.json").chmod(0o600)
    write_json(dest / "diagnostic-rules.json", {
        "elapsed_tick": 8, "remaining_ticks": 248, "kinds": ["gem", "bomb"],
        "construction": "All distinct single-object states from legal spawn/movement prefixes; all legally realizable paddle columns; other spawn slots empty.",
        "rollout": "Suppress future spawns until arrival; gem catches conditional on physically reachable catch; bomb avoidance; record actions/logits.",
        "strata": ["velocity", "time_to_arrival", "reflection_before_arrival"],
        "conflict": {"paddle": 3, "objects": [["bomb", 6, 4, 0], ["gem", 5, 5, 0]]},
        "synthetic_states_excluded": True, "near_tie_relative_tolerance": 1e-6,
        "bootstrap_seed": 12345, "bootstrap_resamples": 10000,
        "bootstrap_percentile_rule": "sorted means at floor(q*N), clamped to N-1",
        "replay_selection": {"median": "sort (return,index), zero-based position 127", "maximum": "largest return, smallest index", "failure": "earliest-index bomb failure"},
    })
    from .probes import probe_manifest
    write_json(dest / "diagnostic-probes.json", probe_manifest())
    (dest / "cli-config.candidate.toml").write_text(
        isolation_config(Path("/private/tmp/ricochet-astra/home/neutral.txt")))
    write_json(dest / "cli-validation-status.json", {
        "status": "pending explicit runtime preflight",
        "requested_model": "gpt-6-astra", "reasoning_effort": "xhigh",
        "reasoning_summary": "detailed", "timeout_seconds": 2700,
        "candidate_settings_are_not_runtime_validation": True,
        "official_documentation": ["https://learn.chatgpt.com/docs/non-interactive-mode",
                                   "https://learn.chatgpt.com/docs/config-file/config-reference"],
        "preview_must_record_difference_from_exec": True,
    })
    write_json(dest / "arithmetic.json", {"numpy": np.__version__, "python": platform.python_version(),
        "machine": platform.machine(), "processor": platform.processor(), "platform": platform.platform(),
        "numpy_configuration": np.show_config(mode="dicts"), "official_dtype": "float32",
        "blas_thread_environment": {name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}})
    (dest / "prompt.txt").write_text(build_prompt(root), encoding="utf-8")
    names = _freeze_names(root)
    provenance = _git_provenance(root, names)
    revision = provenance["head_revision"] if provenance["frozen_files_match_revision"] else None
    preflight = json.loads(Path(preflight_path).read_text()) if preflight_path else None
    if preflight is not None:
        write_json(dest / "runtime-preflight.json", preflight)
    pending_generation = "ricochet-catch generate --execute --run-dir runs/astra-001 --isolated-home /private/tmp/ricochet-astra/home --workdir /private/tmp/ricochet-astra/work --gate runs/preflight/gate-review-template.json --freeze " + str(dest / "manifest.json")
    readiness = {
        "calibration_passed": calibration.get("passed") is True,
        "runtime_preflight": "pending" if preflight is None else preflight.get("runtime_preflight", "requires_review"),
        "astra_generation": "not requested", "hidden_test_evaluation": "not run", "ready_for_generation": False,
        "pending_preflight_invocation": "ricochet-catch preflight --execute --run-dir runs/preflight --isolated-home /private/tmp/ricochet-astra/home --workdir /private/tmp/ricochet-astra/work",
        "pending_generation_invocation": pending_generation,
        "notes": ["Both commands require separate explicit execution; dry runs/tests do not launch them.",
                  "Review the preflight evidence and complete every gate check before generation."]}
    write_json(dest / "readiness.json", readiness)
    manifest = {"format": "ricochet-freeze-v1", "files": {name: sha256_file(root / name) for name in sorted(names)},
        "artifacts": {file.name: sha256_file(file) for file in sorted(dest.iterdir()) if file.is_file()},
        "prompt_sha256": sha256_file(dest / "prompt.txt"), "implementation_revision": revision,
        "implementation_provenance": provenance,
        "calibration_passed": calibration.get("passed") is True,
        "trivial_comparator": calibration.get("trivial_comparator"), "bootstrap_seed": 12345,
        "runtime_preflight": "pending" if preflight is None else preflight.get("runtime_preflight", "requires_review"),
        "astra_generation": "not requested", "hidden_test_evaluation": "not run", "ready_for_generation": False,
        "pending_generation_invocation": pending_generation,
        "readiness_notes": ["Runtime denial/schema smoke requests require separate explicit preflight --execute.",
                            "Review the exact context and all callable capabilities, then complete a hash-bound gate.",
                            "No Astra result exists; hidden test tapes have not been scored."]}
    if previous:
        manifest["previous_freeze"] = {
            "manifest_sha256": sha256_file(previous_path),
            "development_seeds_sha256": previous["artifacts"]["development-seeds.json"],
            "test_seeds_sha256": previous["artifacts"]["test-seeds.private.json"],
            "seed_commitments_reused_unchanged": True,
        }
    write_json(dest / "manifest.json", manifest)
    return manifest
