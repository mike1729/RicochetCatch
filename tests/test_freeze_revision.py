from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from ricochet_catch import freeze
from ricochet_catch.run_astra import RunnerError, sha256_file, write_json


@pytest.fixture
def source_tree(tmp_path):
    source = Path(__file__).resolve().parents[1]
    root = tmp_path / "repo"
    for name in freeze.REQUIRED_FILES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    return root


def git(root, *args):
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "user.name=Freeze tests",
         "-c", "user.email=freeze-tests@example.invalid", "-c", "commit.gpgsign=false", *args],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()


def commit_source(root):
    git(root, "init", "--quiet")
    git(root, "add", ".")
    git(root, "commit", "--quiet", "-m", "Fixture implementation")
    return git(root, "rev-parse", "HEAD")


def test_recursive_fixtures_and_simulator_are_frozen(source_tree, tmp_path):
    fixture_names = ["tests/fixtures/events.jsonl", "tests/fixtures/provenance/README.md",
                     "tests/fixtures/provenance/LICENSE", "tests/.hidden/selectors.data"]
    for name in fixture_names + ["tests/__pycache__/test_stale.pyc", "tests/.pytest_cache/state"]:
        path = source_tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    frozen = tmp_path / "frozen"
    manifest = freeze.freeze_project(source_tree, frozen)
    assert "ricochet_sim.py" in freeze.REQUIRED_FILES
    assert {"ricochet_sim.py", *fixture_names}.issubset(manifest["files"])
    assert not any("__pycache__" in name or ".pytest_cache" in name for name in manifest["files"])
    assert freeze.verify_freeze(source_tree, frozen / "manifest.json") == manifest
    del manifest["files"][fixture_names[0]]
    write_json(frozen / "manifest.json", manifest)
    with pytest.raises(RunnerError, match="omits required"):
        freeze.verify_freeze(source_tree, frozen / "manifest.json")


def test_committed_source_revision_is_exact_and_survives_document_commit(source_tree, tmp_path):
    revision = commit_source(source_tree)
    frozen = tmp_path / "frozen"
    manifest = freeze.freeze_project(source_tree, frozen)
    assert manifest["implementation_revision"] == revision
    assert manifest["implementation_provenance"] == {
        "head_revision": revision, "git_dirty": False,
        "frozen_files_match_revision": True, "uncommitted_frozen_files": [],
    }
    (source_tree / "FREEZE_COMMITMENT.json").write_text('{"commitment":"recorded separately"}\n')
    git(source_tree, "add", "FREEZE_COMMITMENT.json")
    git(source_tree, "commit", "--quiet", "-m", "Record freeze commitment")
    assert git(source_tree, "rev-parse", "HEAD") != revision
    assert freeze.verify_freeze(source_tree, frozen / "manifest.json") == manifest


def test_uncommitted_source_does_not_claim_head_contains_frozen_code(source_tree, tmp_path):
    revision = commit_source(source_tree)
    changed = "src/ricochet_catch/env.py"
    path = source_tree / changed
    path.write_text(path.read_text() + "\n# Uncommitted fixture change.\n")
    added = "tests/fixtures/new.json"
    (source_tree / added).parent.mkdir(parents=True)
    (source_tree / added).write_text("{}\n")
    manifest = freeze.freeze_project(source_tree, tmp_path / "frozen")
    assert manifest["implementation_revision"] is None
    assert manifest["implementation_provenance"] == {
        "head_revision": revision, "git_dirty": True,
        "frozen_files_match_revision": False, "uncommitted_frozen_files": sorted([changed, added]),
    }


def test_unrelated_dirty_document_does_not_hide_exact_source_revision(source_tree, tmp_path):
    revision = commit_source(source_tree)
    (source_tree / "handoff.txt").write_text("Uncommitted handoff, outside the frozen source set.\n")
    manifest = freeze.freeze_project(source_tree, tmp_path / "frozen")
    assert manifest["implementation_revision"] == revision
    assert manifest["implementation_provenance"]["git_dirty"] is True
    assert manifest["implementation_provenance"]["frozen_files_match_revision"] is True


def test_refreeze_reuses_seed_bytes_without_requiring_obsolete_source_hashes(source_tree, tmp_path, monkeypatch):
    calibration = tmp_path / "calibration.json"
    write_json(calibration, {"passed": True, "seeds": list(range(256, 512)), "trivial_comparator": "chase"})
    old = tmp_path / "frozen-v1"
    original = freeze.freeze_project(source_tree, old, calibration)
    original_files = {file.name: file.read_bytes() for file in old.iterdir() if file.is_file()}
    path = source_tree / "src/ricochet_catch/env.py"
    path.write_text(path.read_text() + "\n# Reviewed implementation revision.\n")
    monkeypatch.setattr(freeze.secrets, "randbits", lambda *a: pytest.fail("Hidden seeds were regenerated"))
    new = tmp_path / "frozen-v2"
    manifest = freeze.freeze_project(source_tree, new, previous_freeze_path=old / "manifest.json")
    for name in ("development-seeds.json", "test-seeds.private.json"):
        assert (new / name).read_bytes() == original_files[name]
        assert manifest["artifacts"][name] == original["artifacts"][name]
    assert manifest["calibration_passed"] is True
    assert json.loads((new / "calibration.json").read_text())["seeds"] == list(range(256, 512))
    assert (new / "test-seeds.private.json").stat().st_mode & 0o777 == 0o600
    assert manifest["previous_freeze"] == {
        "manifest_sha256": sha256_file(old / "manifest.json"),
        "development_seeds_sha256": original["artifacts"]["development-seeds.json"],
        "test_seeds_sha256": original["artifacts"]["test-seeds.private.json"],
        "seed_commitments_reused_unchanged": True,
    }
    assert {file.name: file.read_bytes() for file in old.iterdir() if file.is_file()} == original_files
    assert freeze.verify_freeze(source_tree, new / "manifest.json") == manifest


def test_refreeze_verifies_every_previous_artifact_before_creating_destination(source_tree, tmp_path):
    old, new = tmp_path / "frozen-v1", tmp_path / "frozen-v2"
    freeze.freeze_project(source_tree, old)
    (old / "readiness.json").write_text("{}\n")
    with pytest.raises(RunnerError, match="Frozen artifact changed or missing: readiness.json"):
        freeze.freeze_project(source_tree, new, previous_freeze_path=old / "manifest.json")
    assert not new.exists()


def test_refreeze_rejects_changed_development_commitment(source_tree, tmp_path):
    old, new = tmp_path / "frozen-v1", tmp_path / "frozen-v2"
    freeze.freeze_project(source_tree, old)
    calibration = tmp_path / "different-calibration.json"
    write_json(calibration, {"passed": True, "seeds": list(range(256, 512))})
    with pytest.raises(RunnerError, match="preserve the committed development seeds"):
        freeze.freeze_project(source_tree, new, calibration, previous_freeze_path=old / "manifest.json")
    assert not new.exists()


@pytest.mark.parametrize("field,value", [("astra_generation", "completed"), ("hidden_test_evaluation", "scored")])
def test_refreeze_rejects_a_previous_manifest_declaring_generation_or_scoring(source_tree, tmp_path, field, value):
    old, new = tmp_path / "frozen-v1", tmp_path / "frozen-v2"
    manifest = freeze.freeze_project(source_tree, old)
    manifest[field] = value
    write_json(old / "manifest.json", manifest)
    with pytest.raises(RunnerError, match="Only an unevaluated, pre-generation freeze"):
        freeze.freeze_project(source_tree, new, previous_freeze_path=old / "manifest.json")
    assert not new.exists()
