"""Dry-run-first isolated Codex transport. Importing this module never runs Codex.

Runtime preflight is deliberately not inferred from accepted configuration keys.
Only an explicitly reviewed, hash-bound gate can authorize a game request.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any

MODEL = "gpt-6-astra"
TIMEOUT = 45 * 60
PUBLIC_SOURCES = ("env.py", "observations.py", "policy.py")
SCHEMA_PATH = "schemas/tensor-assignments-v2.json"
# All-zero syntax coverage: range/list selectors, preserved list order, both
# value forms and later overlapping writes. This is not a candidate policy.
TRANSPORT_FIXTURE = json.dumps({
    "format": "tensor-assignments-v2", "explanation": "Transport check only.",
    "assignments": [
        {"tensor": "W1", "selectors": [{"start": 0, "stop": 2, "step": 1}, [5, 3]],
         "values": [0, 0, 0, 0]},
        {"tensor": "W1", "selectors": [[1], {"start": 3, "stop": 6, "step": 2}],
         "values": 0},
    ],
}, separators=(",", ":"))
CODE_MODE_UNAVAILABLE_NOTICE = (
    "Code Mode is unavailable because code-mode host is disabled. Code mode will "
    "fail closed; enable `features.code_mode_host` and install `codex-code-mode-host`."
)
DENIAL_PROMPT = (
    "Content-free transport validation. Specifically attempt to calculate 2+2 through "
    "the advertised execution tool. Do not calculate by reasoning as a substitute. "
    "If execution is unavailable, say so. If a call is denied, report the denial. "
    "Do not read or write files, browse, delegate, or invoke other capabilities."
)
SCHEMA_PROMPT = (
    "Content-free structured-output transport validation. Use no tools. Return "
    "exactly this JSON document, unchanged: " + TRANSPORT_FIXTURE
)

# Authored from the public rules, not extracted from the implementing fixtures.
RULES = """Construct the weights of the fixed neural policy below to maximize expected
gem catches. All tensors initially contain zeros. Reason from these mechanics and
source. You have no tools and receive no rollout or performance feedback. Do not
provide a training algorithm or runtime controller. Return exactly the required
JSON object, with a short explanation in its explanation field.

Game rules, version 0.1
The board has 8 columns and 8 rows, indexed from 0 at the left/top to 7. A one-cell
paddle is on row 7, initially column 3. Actions are 0=stay, 1=left, 2=right, moving
0,-1,+1 columns and clipping to [0,7]. Catching a gem awards +1. Catching a bomb
ends the episode with zero reward for that step. Misses award zero. No movement
cost or survival reward exists. Score is gem catches before death or 256 ticks.

An active object is (kind,row,column,vx), with kind gem/bomb and vx in {-1,0,+1}.
Vertical velocity is always +1 row each tick. An episode begins with no objects
and 256 remaining ticks. Independently for each tick, spawn probability is .45;
a spawn is a bomb with probability .25, its column is uniform over 0..7, and its
vx is uniform over -1,0,+1. The complete tape is generated before play with four
random draws each tick, including ticks with no spawn; it does not depend on
actions. Individual seeds, future arrivals and RNG state are hidden. There is at
most one spawn per tick, so active objects occupy distinct rows.

Each step first moves the paddle. Every active object then simultaneously tries
next_column=column+vx. If this is outside [0,7], reverse vx persistently and use
next_column=column+vx with that reversed vx. Advance row by 1. Objects reaching
row 7 collide only at their updated column with the new paddle column: award a
matching gem, terminate on a matching bomb, otherwise miss; remove each arriving
object. Decrement remaining time once. On a bomb collision return terminated
immediately, including on the last tick, with no spawn. Otherwise if time reaches
zero return truncated with no spawn. Otherwise append this tick's tape entry at
row 0 if present. A new spawn first moves on the following tick. Merely reaching
a wall does not reverse vx; only an attempted outward move reverses it. There
are no collisions above row 7, swept-path collisions, or object-object collisions.
Actions must be integer indices, excluding booleans and floats. Stepping after
termination/truncation is an error.

Observation and network
The observation is float32, length 449. Its first 448 entries are an 8x8x7 binary
grid in row,column,channel order. Index=((row*8+column)*7+channel). Channels are:
0=paddle; 1,2,3=gem with vx=-1,0,+1; 4,5,6=bomb with vx=-1,0,+1. Paddle occupies
(7,paddle_column,0), and each object exactly one cell/channel. All other entries
are zero. Entry 448 is remaining_ticks/256. No preprocessing, frame stacking,
computed targets or persistent memory is available. The full padded input stays.

Use column-vector convention and float32 stored tensors and arithmetic:
W1:(32,449), b1:(32,), W2:(32,32), b2:(32,), W3:(3,32), b3:(3,).
h1=tanh(W1@x+b1); h2=tanh(W2@h1+b2); z=W3@h2+b3.
Use NumPy 2.3.4 tanh. Action=argmax(z), ties selecting the smallest action index.
There are exactly 15,555 parameters. No recurrence, code, masking, sampling or
weight updates may influence inference. Nonfinite inputs/logits are errors.

The network is evaluated on 256 independently seeded held-out episodes from the
distribution above, at most 256 ticks each, using deterministic actions. The
display may draw diamonds, crosses, a paddle bar and directional arrows at 5 ticks
per second, but only the exact observation above reaches the policy.

Output contract: tensor-assignments-v2
Return bare JSON with exactly format, explanation, assignments. format must be
"tensor-assignments-v2"; explanation is text of at most 10,000 characters.
assignments is an array of at most 10,000 entries. Each entry has exactly tensor,
selectors, values. tensor is W1,b1,W2,b2,W3,b3. Supply one selector per dimension.
A selector is either a nonempty ordered list of distinct integer indices in
[0,n), or exactly {"start":i,"stop":j,"step":k} with 0<=i<j<=n, 1<=k<=n and
half-open range semantics. Negative indices, booleans and empty selections are
invalid. Preserve list order. values is a finite JSON number broadcast over the
selection, or a flat list with the Cartesian product's length, last axis varying
fastest. For selectors [[2,0],[7,0]], positions are (2,7),(2,0),(0,7),(0,0).
Apply assignments in array order; later writes overwrite earlier writes. Omitted
entries stay zero. Empty assignments are valid. All supplied values, including
overwritten values, must have magnitude <=10,000 and remain finite in float32.
Maximum response size is 5 MiB and total scalar writes including overwrites is
1,000,000. Duplicate JSON keys, unknown fields and trailing prose are invalid.
Only an optional single outer Markdown fence opening with ```json and closing
with ``` is accepted; only whitespace may surround the document/fence. No repair
or partial application occurs. The structural JSON Schema follows; semantic
validation additionally enforces tensor bounds, counts and float32 conversion.

Neutral syntax illustration using a DIFFERENT TOY SCHEMA (not a policy, not valid
for submission): toy tensor Q has shape (3,8), and a toy assignment
{"tensor":"Q","selectors":[[2,0],[7,0]],"values":[1,2,3,4]} writes Q[2,7]=1,
Q[2,0]=2,Q[0,7]=3,Q[0,0]=4. In the real observation, row r/channel ch columns
have indices 56*r+ch+7*c for c=0..7. This describes serialization only.

No tools, execution, training, rollouts, search, delegation or feedback are
permitted. Submit numerical assignments and explanatory text only.
"""


class RunnerError(ValueError):
    """A missing or invalid protocol prerequisite; no retry is implied."""


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_prompt(root: str | Path) -> str:
    """Include only authored public rules, exact public mechanics and schema."""
    root = Path(root)
    pieces = [RULES]
    for name in PUBLIC_SOURCES:
        source = root / "src" / "ricochet_catch" / name
        if not source.is_file():
            raise RunnerError(f"Missing public source: {source}")
        pieces.append(f"\nExact source: {name}\n```python\n{source.read_text(encoding='utf-8')}\n```\n")
    schema = (root / SCHEMA_PATH).read_text(encoding="utf-8")
    pieces.append(f"\nStructural output schema\n```json\n{schema}\n```\n")
    return "\n".join(pieces)


def audit_events(raw: str) -> dict[str, Any]:
    """Conservative JSONL audit, distinguishing notices from actual call items.

    These known events follow the documented exec stream. New schemas are held
    for inspection instead of silently treated as non-tool events.
    """
    known_events = {"thread.started", "turn.started", "turn.completed", "turn.failed", "error"}
    known_items = {"agent_message", "reasoning", "plan", "todo_list", "error"}
    call_items = {"command_execution", "file_change", "mcp_tool_call", "web_search", "tool_call",
                  "function_call", "computer_call", "image_generation", "collab_tool_call",
                  "collaboration_tool_call", "delegation", "code_execution"}
    calls, unknown, compaction, errors, notices = [], [], [], [], []
    completed = False
    turn_started = False
    count = 0
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        count += 1
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            unknown.append({"line": number, "reason": "invalid JSON"})
            continue
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            unknown.append({"line": number, "reason": "missing event type"})
            continue
        kind = event["type"]
        if completed:
            errors.append({"line": number, "type": kind, "reason": "event after completed turn"})
        if "compact" in kind.lower():
            compaction.append({"line": number, "type": kind})
        elif kind in {"item.started", "item.updated", "item.completed"}:
            item = event.get("item")
            if not isinstance(item, dict):
                unknown.append({"line": number, "reason": "missing item object"})
                continue
            item_type = item.get("type")
            if isinstance(item_type, str) and "compact" in item_type.lower():
                compaction.append({"line": number, "type": item_type})
            elif item_type in call_items:
                # A denied attempt still violates the game's no-tools condition.
                calls.append({"line": number, "type": item_type, "id": item.get("id"),
                              "status": item.get("status")})
            elif item_type not in known_items:
                unknown.append({"line": number, "reason": "unknown item", "type": item_type})
            elif item_type == "error":
                # Exact envelope/message observed in the authors' published
                # Breakout stream. Only this pre-turn notice is allowlisted;
                # it does not prove execution denial or validate any tool stub.
                if (kind == "item.completed" and not turn_started
                        and item.get("message") == CODE_MODE_UNAVAILABLE_NOTICE):
                    notices.append({"line": number, "type": item_type,
                                    "message": item["message"]})
                else:
                    errors.append({"line": number, "type": item_type})
        elif kind not in known_events:
            unknown.append({"line": number, "reason": "unknown event", "type": kind})
        if kind == "turn.completed":
            if not turn_started:
                errors.append({"line": number, "type": kind, "reason": "completion without turn start"})
            completed = True
        if kind == "turn.started":
            if turn_started:
                errors.append({"line": number, "type": kind, "reason": "multiple turns in a fresh single-turn run"})
            turn_started = True
        if kind in {"turn.failed", "error"}:
            errors.append({"line": number, "type": kind})
    valid = completed and not (calls or unknown or compaction or errors)
    return {"valid": valid, "completed": completed, "events": count, "tool_calls": calls,
            "unknown_events": unknown, "compaction_events": compaction, "errors": errors,
            "startup_notices": notices, "evidence_scope": "Observed local JSONL stream only."}


DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "unified_exec_tty", "code_mode", "code_mode_host",
    "code_mode_only", "code_mode_interrupt", "code_mode_prewarm", "code_mode_tool_search",
    "apps", "enable_mcp_apps", "plugins", "remote_plugin", "plugin_sharing", "hooks",
    "multi_agent", "multi_agent_v2", "multi_agent_v2_dynamic_tools", "agent_message_board",
    "browser_use", "browser_use_external", "browser_use_full_cdp_access", "computer_use",
    "image_generation", "view_image", "in_app_browser", "memories", "chronicle",
    "skill_search", "skill_mcp_dependency_install", "shell_snapshot", "sleep_tool",
    "workspace_dependencies", "goals", "tool_suggest", "standalone_web_search",
    "unbounded_connection_retries",
)


def isolation_config(neutral_path: Path) -> str:
    """Candidate supported settings; acceptance never establishes tool denial."""
    base = [
        f'model = "{MODEL}"', 'approval_policy = "never"', 'sandbox_mode = "read-only"',
        'model_reasoning_effort = "xhigh"', 'model_reasoning_summary = "detailed"',
        f'model_instructions_file = {json.dumps(str(neutral_path))}',
        'project_doc_max_bytes = 0', 'project_doc_fallback_filenames = []',
        'web_search = "disabled"', 'skills.bundled.enabled = false',
        'skills.include_instructions = false', 'shell_environment_policy.inherit = "none"',
        'apps._default.enabled = false', 'agents.enabled = false', 'mcp_servers = {}',
        'features.skip_host_skill_discovery = true',
    ]
    base.extend(f'features.{name} = false' for name in DISABLED_FEATURES)
    return "\n".join(base) + "\n"


def _check_locations(root: Path, home: Path, work: Path) -> None:
    normal_homes = {Path.home().joinpath(".codex").resolve(),
                    Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).resolve()}
    related = lambda one, two: one == two or one in two.parents or two in one.parents
    if any(related(home, normal) for normal in normal_homes) or related(root, home) or related(root, work):
        raise RunnerError("Use a dedicated Codex home and an empty work directory outside the repository.")
    if home == work or home in work.parents or work in home.parents:
        raise RunnerError("Isolated home and clean working directory must be separate.")
    if work.exists() and any(work.iterdir()):
        raise RunnerError("Model working directory must be empty.")


def _arguments(work: Path, schema: Path | None, response: Path) -> list[str]:
    argv = ["codex", "exec", "--model", MODEL, "--ephemeral", "--json", "--strict-config",
            "--skip-git-repo-check", "--ignore-rules", "--sandbox", "read-only", "--cd", str(work),
            "-c", 'approval_policy="never"', "-c", 'model_reasoning_effort="xhigh"',
            "-c", 'model_reasoning_summary="detailed"', "--output-last-message", str(response)]
    if schema is not None:
        argv.extend(["--output-schema", str(schema)])
    return argv + ["-"]


def _fingerprint(home: Path, work: Path, config: str, neutral: bytes, schema: bytes,
                 cli_version: str) -> str:
    body = {"home": str(home), "work": str(work), "config": config,
            "neutral_sha256": hashlib.sha256(neutral).hexdigest(),
            "schema_sha256": hashlib.sha256(schema).hexdigest(), "model": MODEL,
            "cli_version": cli_version, "runner_sha256": sha256_file(__file__)}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _initialize_isolation(home: Path, work: Path, config: str, neutral: bytes) -> None:
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    home.chmod(0o700)
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Never inspect, copy or archive authentication contents. A supported login
    # into this isolated home is an independent user action if needed.
    for name, expected in (("config.toml", config.encode()), ("neutral.txt", neutral)):
        target = home / name
        if target.exists() and target.read_bytes() != expected:
            raise RunnerError(f"Existing isolated {name} differs; preserve it and use a new context.")
        if not target.exists():
            target.write_bytes(expected)
            target.chmod(0o600)
    forbidden = {"AGENTS.md", "AGENTS.override.md", "skills", "memories", "plugins", "hooks", ".agents"}
    if forbidden.intersection(p.name for p in home.iterdir()):
        raise RunnerError("Isolated home contains discovered instructions/capabilities; use a new context.")


def _child_environment(home: Path) -> dict[str, str]:
    # No token/key environment values are read or persisted. Login credentials
    # must be managed by Codex's supported login flow for this selected home.
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR") if key in os.environ}
    env["CODEX_HOME"] = str(home)
    return env


def _execute(argv: list[str], prompt: Path, dest: Path, env: dict[str, str],
             work: Path, timeout: int) -> dict[str, Any]:
    """Exactly one process; preserve streams even on timeout or interrupt."""
    started = datetime.now(timezone.utc).isoformat()
    result: dict[str, Any] = {"started_at": started, "status": "started"}
    write_json(dest / "execution.json", result)
    with prompt.open("rb") as stdin, (dest / "events.jsonl").open("wb") as stdout, (dest / "stderr.txt").open("wb") as stderr:
        try:
            proc = subprocess.run(argv, stdin=stdin, stdout=stdout, stderr=stderr, cwd=work,
                                  env=env, timeout=timeout, check=False)
            result.update(status="finished", exit_status=proc.returncode)
        except subprocess.TimeoutExpired:
            result.update(status="timeout", exit_status=None)
        except KeyboardInterrupt:
            result.update(status="interrupted", exit_status=None)
        except OSError as exc:
            result.update(status="startup_error", exit_status=None, error=str(exc))
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    audit = audit_events((dest / "events.jsonl").read_text(encoding="utf-8", errors="replace"))
    write_json(dest / "execution.json", result)
    write_json(dest / "audit.json", audit)
    return {"execution": result, "audit": audit}


def _offline_inspect(dest: Path, env: dict[str, str], work: Path) -> str:
    commands = {"version": ["codex", "--version"], "exec-help": ["codex", "exec", "--help"],
                "features": ["codex", "features", "list"],
                "preview": ["codex", "debug", "prompt-input", "Content-free offline context inspection."]}
    statuses = {}
    for name, argv in commands.items():
        try:
            proc = subprocess.run(argv, cwd=work, env=env, capture_output=True, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RunnerError(f"Offline CLI inspection failed: {name}: {exc}") from exc
        (dest / f"{name}.stdout.txt").write_bytes(proc.stdout)
        (dest / f"{name}.stderr.txt").write_bytes(proc.stderr)
        statuses[name] = {"argv": argv, "exit_status": proc.returncode}
        if proc.returncode != 0:
            write_json(dest / "offline-inspection.json", statuses)
            raise RunnerError(f"Offline CLI inspection failed: {name}; inspect saved output.")
    statuses["preview_differences"] = (
        "debug prompt-input shares the selected home/config and cwd, but lacks exec's strict-config, "
        "ignore-rules, output-schema and ephemeral flags. This preview does not verify the real exec path."
    )
    write_json(dest / "offline-inspection.json", statuses)
    return (dest / "version.stdout.txt").read_text().strip()


def _check_gate(path: Path | None, fingerprint: str) -> dict[str, Any]:
    if path is None or not path.is_file():
        raise RunnerError("Runtime preflight pending: provide a reviewed gate before any game request.")
    gate = json.loads(path.read_text())
    checks = ("execution_unavailable_or_explicitly_denied", "schema_transport_valid",
              "instructions_reviewed", "effective_declarations_reviewed", "managed_configuration_reviewed",
              "no_unverified_capabilities", "actual_exec_route_reviewed")
    if gate.get("fingerprint") != fingerprint or not all(gate.get(key) is True for key in checks):
        raise RunnerError("Runtime gate is incomplete or does not match the exact current isolation settings/version.")
    if not gate.get("reviewer") or not gate.get("reviewed_at") or not gate.get("evidence"):
        raise RunnerError("Runtime gate requires an identified reviewer, date and archived evidence hashes.")
    for evidence in gate["evidence"]:
        evidence_path = (path.parent / evidence["path"]).resolve()
        if not evidence_path.is_relative_to(path.parent.resolve()):
            raise RunnerError("Gate evidence must be archived inside the preflight directory.")
        if not evidence_path.is_file() or sha256_file(evidence_path) != evidence["sha256"]:
            raise RunnerError("Runtime gate evidence is missing or changed.")
    return gate


def prepare_run(root: str | Path, run_dir: str | Path, isolated_home: str | Path,
                clean_workdir: str | Path, *, mode: str = "generate", execute: bool = False,
                gate_path: str | Path | None = None, freeze_path: str | Path | None = None,
                timeout: int = TIMEOUT) -> dict[str, Any]:
    """Prepare artifacts by default; explicit execution never retries a request.

    Preflight execution runs two content-free requests and creates an incomplete
    review template. It cannot automatically certify denial or hidden behavior.
    """
    if mode not in {"generate", "preflight"} or type(timeout) is not int or timeout <= 0:
        raise RunnerError("Invalid run mode or timeout.")
    root, dest = Path(root).resolve(), Path(run_dir).resolve()
    home, work = Path(isolated_home).resolve(), Path(clean_workdir).resolve()
    _check_locations(root, home, work)
    if any(dest == place or dest in place.parents or place in dest.parents for place in (home, work)):
        raise RunnerError("Run artifacts must be separate from the isolated home and clean working directory.")
    if dest.exists():
        raise RunnerError("Run directory already exists; attempts and dry runs are never overwritten.")
    dest.mkdir(parents=True, mode=0o700)
    neutral = (root / "prompts/neutral.txt").read_bytes()
    schema = (root / SCHEMA_PATH).read_bytes()
    (dest / "neutral.txt").write_bytes(neutral)
    (dest / "schema.json").write_bytes(schema)
    config = isolation_config(home / "neutral.txt")
    (dest / "config.toml").write_text(config)
    prompts = {"generation": build_prompt(root)} if mode == "generate" else {
        "execution-denial": DENIAL_PROMPT, "schema-transport": SCHEMA_PROMPT}
    invocations = {}
    for name, prompt in prompts.items():
        folder = dest if mode == "generate" else dest / name
        folder.mkdir(exist_ok=True)
        (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
        argv = _arguments(work, dest / "schema.json" if name != "execution-denial" else None, folder / "response.json")
        invocations[name] = {"argv": argv, "stdin": str(folder / "prompt.txt"),
                             "stdout": str(folder / "events.jsonl"), "stderr": str(folder / "stderr.txt"),
                             "environment": {"CODEX_HOME": str(home)},
                             "display": shlex.join(argv)}
    manifest: dict[str, Any] = {"mode": mode, "dry_run": not execute, "requested_model": MODEL,
        "reasoning_effort": "xhigh", "reasoning_summary": "detailed", "timeout_seconds": timeout,
        "isolated_home": str(home), "working_directory": str(work), "invocations": invocations,
        "prompt_hashes": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in prompts.items()},
        "runtime_preflight": "pending", "experimental_requests": 0, "smoke_requests": 0,
        "notes": ["Candidate settings require runtime validation; dry run issues no CLI or model requests.",
                  "No credentials are copied, inspected or archived. Login separately in the isolated home if needed."]}
    write_json(dest / "manifest.json", manifest)
    if not execute:
        return manifest
    _initialize_isolation(home, work, config, neutral)
    env = _child_environment(home)
    version = _offline_inspect(dest, env, work)
    fingerprint = _fingerprint(home, work, config, neutral, schema, version)
    manifest.update(cli_version=version, fingerprint=fingerprint)
    if mode == "generate":
        original_gate_path = Path(gate_path).resolve() if gate_path else None
        gate = _check_gate(original_gate_path, fingerprint)
        if freeze_path is None:
            raise RunnerError("A frozen implementation manifest is required before generation.")
        from .freeze import verify_freeze
        frozen = verify_freeze(root, Path(freeze_path))
        if not frozen.get("calibration_passed"):
            raise RunnerError("Frozen development calibration has not passed.")
        if frozen["prompt_sha256"] != manifest["prompt_hashes"]["generation"]:
            raise RunnerError("Prepared game prompt differs from the frozen prompt.")
        archived_gate = dict(gate)
        archived_gate["evidence"] = []
        (dest / "gate-evidence").mkdir()
        for index, item in enumerate(gate["evidence"]):
            relative = f"gate-evidence/{index}-{Path(item['path']).name}"
            shutil.copyfile(original_gate_path.parent / item["path"], dest / relative)
            archived_gate["evidence"].append({"path": relative, "sha256": item["sha256"]})
        write_json(dest / "runtime-gate.json", archived_gate)
        shutil.copyfile(Path(freeze_path), dest / "frozen-manifest.json")
        manifest.update(runtime_gate_sha256=sha256_file(dest / "runtime-gate.json"),
                        frozen_manifest_sha256=sha256_file(dest / "frozen-manifest.json"),
                        runtime_preflight="reviewed gate supplied")
        manifest["experimental_requests"] = 1
        # Mark the attempt before starting a process, including failure paths.
        write_json(dest / "manifest.json", manifest)
        item = invocations["generation"]
        manifest["result"] = _execute(item["argv"], dest / "prompt.txt", dest, env, work, timeout)
        manifest["output_hashes"] = {name: sha256_file(dest / name) for name in
            ("events.jsonl", "stderr.txt", "response.json", "audit.json", "execution.json") if (dest / name).is_file()}
    else:
        manifest["results"] = {}
        for name, item in invocations.items():
            manifest["smoke_requests"] += 1
            write_json(dest / "manifest.json", manifest)
            manifest["results"][name] = _execute(item["argv"], Path(item["stdin"]), dest / name, env, work, timeout)
        response = dest / "schema-transport/response.json"
        transport_valid = False
        try:
            from .assignments import parse_assignments
            from jsonschema import validate, ValidationError
            document = json.loads(response.read_text())
            if document == json.loads(TRANSPORT_FIXTURE):
                validate(document, json.loads(schema))
                parse_assignments(response.read_bytes())
                transport_valid = True
        except (ValueError, OSError, ValidationError):
            pass
        template = {"fingerprint": fingerprint, "reviewer": "", "reviewed_at": "", "evidence": [],
                    "execution_unavailable_or_explicitly_denied": False,
                    "schema_transport_valid": transport_valid, "instructions_reviewed": False,
                    "effective_declarations_reviewed": False, "managed_configuration_reviewed": False,
                    "no_unverified_capabilities": False, "actual_exec_route_reviewed": False,
                    "notes": "Review actual call/denial evidence and every declaration. Ordinary answers, refusals and startup notices alone do not validate reachable stubs."}
        write_json(dest / "gate-review-template.json", template)
        manifest["runtime_preflight"] = "requires_manual_review"
    write_json(dest / "manifest.json", manifest)
    return manifest


def validate_generation_run(root: str | Path, run_dir: str | Path,
                            freeze_path: str | Path) -> dict[str, Any]:
    """Check local provenance, requiring unchanged raw bytes before parsing."""
    from .freeze import verify_freeze
    root, dest = Path(root).resolve(), Path(run_dir).resolve()
    frozen = verify_freeze(root, freeze_path)
    manifest = json.loads((dest / "manifest.json").read_text())
    if manifest.get("dry_run") is not False or manifest.get("mode") != "generate" or manifest.get("experimental_requests") != 1:
        raise RunnerError("Official evaluation requires one explicitly executed game generation.")
    if not frozen.get("calibration_passed"):
        raise RunnerError("Official evaluation requires a passing frozen development calibration.")
    if manifest.get("frozen_manifest_sha256") != sha256_file(freeze_path):
        raise RunnerError("Run and supplied freeze manifest differ.")
    if sha256_file(dest / "frozen-manifest.json") != manifest.get("frozen_manifest_sha256"):
        raise RunnerError("Archived freeze manifest changed.")
    if sha256_file(dest / "runtime-gate.json") != manifest.get("runtime_gate_sha256"):
        raise RunnerError("Archived runtime gate changed.")
    _check_gate(dest / "runtime-gate.json", manifest.get("fingerprint"))
    for name in ("response.json", "events.jsonl", "stderr.txt", "audit.json", "execution.json"):
        if not (dest / name).is_file() or sha256_file(dest / name) != manifest.get("output_hashes", {}).get(name):
            raise RunnerError(f"Generation output missing or changed: {name}")
    if sha256_file(dest / "prompt.txt") != frozen["prompt_sha256"]:
        raise RunnerError("Generated prompt was not the frozen prompt.")
    result = json.loads((dest / "execution.json").read_text())
    if result.get("status") != "finished" or result.get("exit_status") != 0:
        raise RunnerError("Generation did not finish successfully.")
    stream = (dest / "events.jsonl").read_text()
    audit = audit_events(stream)
    if not audit["valid"]:
        raise RunnerError("Generation audit failed: tools, unknown events, compaction or incomplete/error stream.")
    usage, returned_model = None, None
    for line in stream.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if isinstance(event.get("model"), str):
            returned_model = event["model"]
        if event["type"] == "turn.completed":
            usage = event.get("usage")
    return {"valid": True, "audit_valid": True, "audit": audit,
            "response_path": str(dest / "response.json"), "response_sha256": sha256_file(dest / "response.json"),
            "requested_model": manifest["requested_model"], "cli_version": manifest.get("cli_version"),
            "reasoning_effort": manifest.get("reasoning_effort"), "fingerprint": manifest.get("fingerprint"),
            "returned_model": returned_model, "usage": usage, "execution": result,
            "attempts": 1, "evidence_scope": "Observed local artifacts; parse unchanged response before scoring."}
