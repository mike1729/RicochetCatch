from __future__ import annotations

import json
import re
from pathlib import Path
import shutil
import subprocess

from jsonschema import validate
import numpy as np
import pytest

from ricochet_catch import run_astra as runner
from ricochet_catch.assignments import parse_assignments
from ricochet_catch.freeze import REQUIRED_FILES, freeze_project, verify_freeze

AUTHOR_FIXTURE = Path(__file__).parent / "fixtures/authors-breakout"


@pytest.fixture
def repository(tmp_path):
    source = Path(__file__).resolve().parents[1]
    root = tmp_path / "repo"
    for name in REQUIRED_FILES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    return root


def events(*extra):
    rows = [{"type": "thread.started", "thread_id": "fixture"}, {"type": "turn.started"}]
    rows.extend(extra)
    rows.extend([{"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}},
                 {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}])
    return "\n".join(json.dumps(row) for row in rows)


def test_dry_run_never_invokes_cli_or_creates_isolation(repository, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run attempted a subprocess")
    monkeypatch.setattr(subprocess, "run", forbidden)
    home, work = tmp_path / "home", tmp_path / "work"
    result = runner.prepare_run(repository, tmp_path / "run", home, work)
    assert result["dry_run"] and result["experimental_requests"] == 0
    assert result["runtime_preflight"] == "pending"
    assert not home.exists() and not work.exists()
    invocation = result["invocations"]["generation"]
    assert invocation["argv"][0:4] == ["codex", "exec", "--model", "gpt-6-astra"]
    assert "--strict-config" in invocation["argv"]
    assert "--ignore-user-config" not in invocation["argv"]
    assert invocation["argv"][-1] == "-"
    with pytest.raises(runner.RunnerError, match="already exists"):
        runner.prepare_run(repository, tmp_path / "run", home, work)


def test_preflight_prompts_are_content_free(repository, tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("CLI called"))
    result = runner.prepare_run(repository, tmp_path / "preflight", tmp_path / "home", tmp_path / "work", mode="preflight")
    assert result["smoke_requests"] == 0
    assert set(result["invocations"]) == {"execution-denial", "schema-transport"}
    for name in result["invocations"]:
        prompt = Path(result["invocations"][name]["stdin"]).read_text()
        for forbidden in ("gem", "bomb", "row", "449", "predictive", "spawn", "seed"):
            assert not re.search(r"\b" + forbidden + r"\b", prompt.lower())
    assert "2+2" in runner.DENIAL_PROMPT


def test_schema_transport_exercises_nonempty_assignment_syntax_without_weights():
    root = Path(__file__).resolve().parents[1]
    document = json.loads(runner.TRANSPORT_FIXTURE)
    validate(document, json.loads((root / runner.SCHEMA_PATH).read_text()))
    first, second = document["assignments"]
    assert first["selectors"] == [{"start": 0, "stop": 2, "step": 1}, [5, 3]]
    assert first["values"] == [0, 0, 0, 0]
    assert second["selectors"] == [[1], {"start": 3, "stop": 6, "step": 2}]
    assert second["values"] == 0
    result = parse_assignments(runner.TRANSPORT_FIXTURE)
    assert result.expanded_scalar_writes == 6
    assert result.unique_entries == 4 and result.overwritten_entries == 2
    assert all(np.count_nonzero(tensor) == 0 for tensor in result.tensors.values())


def test_isolation_disables_agent_tools_and_uses_authors_neutral_instruction(repository):
    config = runner.isolation_config(Path("/isolated/neutral.txt"))
    assert 'agents.enabled = false\n' in config
    assert (repository / "prompts/neutral.txt").read_bytes() == b"Answer the user's request.\n"


def test_prompt_uses_only_allowlisted_source_and_authored_rules(repository):
    (repository / "GAME_DESIGN.md").write_text("DO_NOT_LEAK_FIXTURE_OR_SCORE")
    (repository / "EXPERIMENT_PLAN.md").write_text("DO_NOT_LEAK_PRIVATE_PLAN")
    for name in ("baselines.py", "probes.py", "evaluate.py"):
        (repository / "src/ricochet_catch" / name).write_text("DO_NOT_LEAK_CONTROLLER")
    (repository / "src/ricochet_catch/env.py").write_text("PUBLIC_ENV_SOURCE_EXACT")
    prompt = runner.build_prompt(repository)
    assert "PUBLIC_ENV_SOURCE_EXACT" in prompt
    assert "DO_NOT_LEAK" not in prompt
    assert "15,555" in prompt and "256 independently seeded held-out episodes" in prompt
    assert "DIFFERENT TOY SCHEMA" in prompt
    assert "tensor-assignments-v2" in prompt


@pytest.mark.parametrize("item", ["command_execution", "mcp_tool_call", "file_change", "web_search", "collaboration_tool_call"])
@pytest.mark.parametrize("status", ["completed", "denied", "failed"])
def test_any_actual_tool_attempt_invalidates_primary_condition(item, status):
    report = runner.audit_events(events({"type": "item.completed", "item": {"type": item, "id": "1", "status": status}}))
    assert not report["valid"]
    assert report["tool_calls"][0]["status"] == status


def test_event_audit_is_conservative_about_unknown_compaction_and_incomplete():
    assert runner.audit_events(events())["valid"]
    assert not runner.audit_events('{"type":"turn.started"}')["valid"]
    assert runner.audit_events(events({"type": "item.started", "item": {"type": "new_capability"}}))["unknown_events"]
    assert runner.audit_events(events({"type": "context.compacted"}))["compaction_events"]
    assert runner.audit_events(events({"type": "item.completed", "item": {"type": "compaction"}}))["compaction_events"]
    assert runner.audit_events("not json")["unknown_events"]
    assert not runner.audit_events(events({"type": "error", "message": "bad"}))["valid"]


@pytest.mark.parametrize("raw", [
    '{"type":"turn.completed"}',
    '{"type":"turn.completed"}\n{"type":"turn.started"}',
    events({"type": "turn.started"}),
    events() + '\n{"type":"turn.completed"}',
    events() + '\n{"type":"item.completed","item":{"type":"agent_message","text":"late"}}',
])
def test_event_audit_requires_one_started_and_completed_turn(raw):
    report = runner.audit_events(raw)
    assert not report['valid'] and report['errors']


def test_authors_real_breakout_stream_preserves_provenance_and_passes_audit():
    path = AUTHOR_FIXTURE / "events.jsonl"
    provenance = json.loads((AUTHOR_FIXTURE / "provenance.json").read_text())
    assert path.stat().st_size == provenance["bytes"] == 10_977
    assert runner.sha256_file(path) == provenance["sha256"] == "39ab8479e5f2afac77688634ea3fc54bea6a1f0d41a59eeb97d39b5b65ea3270"
    report = runner.audit_events(path.read_text())
    assert report["valid"] and report["completed"]
    assert not report["tool_calls"] and not report["errors"]
    assert report["startup_notices"] == [{"line": 2, "type": "error", "message": runner.CODE_MODE_UNAVAILABLE_NOTICE}]


def test_generation_validator_accepts_authors_real_tool_free_event_stream(repository, tmp_path):
    # Construct local provenance only: no CLI, model request or policy scoring.
    calibration = tmp_path / "calibration.json"
    runner.write_json(calibration, {"passed": True, "seeds": list(range(256))})
    frozen = tmp_path / "frozen"
    freeze_project(repository, frozen, calibration)
    dest = tmp_path / "generation-fixture"
    manifest = runner.prepare_run(repository, dest, tmp_path / "home", tmp_path / "work")
    shutil.copyfile(AUTHOR_FIXTURE / "events.jsonl", dest / "events.jsonl")
    shutil.copyfile(frozen / "manifest.json", dest / "frozen-manifest.json")
    (dest / "response.json").write_text("{}")
    (dest / "stderr.txt").write_text("")
    runner.write_json(dest / "execution.json", {"status": "finished", "exit_status": 0})
    runner.write_json(dest / "audit.json", runner.audit_events((dest / "events.jsonl").read_text()))
    gate = {key: True for key in ("execution_unavailable_or_explicitly_denied", "schema_transport_valid",
        "instructions_reviewed", "effective_declarations_reviewed", "managed_configuration_reviewed",
        "no_unverified_capabilities", "actual_exec_route_reviewed")}
    gate.update(fingerprint="local-validator-fixture", reviewer="Fixture auditor", reviewed_at="2026-11-03",
                evidence=[{"path": "events.jsonl", "sha256": runner.sha256_file(dest / "events.jsonl")}])
    runner.write_json(dest / "runtime-gate.json", gate)
    manifest.update(dry_run=False, experimental_requests=1, fingerprint=gate["fingerprint"],
                    runtime_gate_sha256=runner.sha256_file(dest / "runtime-gate.json"),
                    frozen_manifest_sha256=runner.sha256_file(frozen / "manifest.json"),
                    output_hashes={name: runner.sha256_file(dest / name) for name in
                        ("response.json", "events.jsonl", "stderr.txt", "audit.json", "execution.json")})
    runner.write_json(dest / "manifest.json", manifest)
    assert runner.validate_generation_run(repository, dest, frozen / "manifest.json")["audit_valid"]


@pytest.mark.parametrize("message", [
    "Code Mode unavailable", runner.CODE_MODE_UNAVAILABLE_NOTICE + " ",
    runner.CODE_MODE_UNAVAILABLE_NOTICE.lower(), "Reconnecting... 1/5",
])
def test_nonmatching_error_items_remain_fatal_even_before_turn(message):
    startup = json.dumps({"type": "item.completed", "item": {"type": "error", "message": message}})
    report = runner.audit_events(startup + "\n" + events())
    assert not report["valid"] and report["errors"]


@pytest.mark.parametrize("event_type", ["item.started", "item.updated", "error", "startup.notice"])
def test_matching_message_in_unobserved_envelope_is_not_allowlisted(event_type):
    notice = {"type": event_type, "item": {"type": "error", "message": runner.CODE_MODE_UNAVAILABLE_NOTICE},
              "message": runner.CODE_MODE_UNAVAILABLE_NOTICE}
    report = runner.audit_events(json.dumps(notice) + "\n" + events())
    assert not report["valid"]
    assert report["errors"] or report["unknown_events"]


def test_exact_notice_after_turn_started_remains_fatal():
    notice = {"type": "item.completed", "item": {"type": "error", "message": runner.CODE_MODE_UNAVAILABLE_NOTICE}}
    report = runner.audit_events(events(notice))
    assert not report["valid"] and report["errors"] and not report["startup_notices"]


def test_startup_notice_without_completed_turn_does_not_validate_a_run():
    notice = {"type": "item.completed", "item": {"type": "error", "message": runner.CODE_MODE_UNAVAILABLE_NOTICE}}
    report = runner.audit_events(json.dumps(notice))
    assert not report["valid"] and report["startup_notices"]


def test_locations_and_dirty_workdir_fail_before_execution(repository, tmp_path):
    with pytest.raises(runner.RunnerError, match="outside"):
        runner.prepare_run(repository, tmp_path / "run", repository / "home", tmp_path / "work")
    work = tmp_path / "work"
    work.mkdir()
    (work / "source.py").write_text("secret")
    with pytest.raises(runner.RunnerError, match="empty"):
        runner.prepare_run(repository, tmp_path / "run", tmp_path / "home", work)


def test_normal_home_hierarchy_and_artifact_overlap_are_rejected(repository, tmp_path, monkeypatch):
    normal_home = tmp_path / "normal-codex"
    monkeypatch.setenv("CODEX_HOME", str(normal_home))
    for proposed in (normal_home, normal_home / "child", normal_home.parent):
        with pytest.raises(runner.RunnerError, match="dedicated"):
            runner.prepare_run(repository, tmp_path / "run", proposed, tmp_path / "work")
    with pytest.raises(runner.RunnerError, match="separate"):
        runner.prepare_run(repository, tmp_path / "work/run", tmp_path / "home", tmp_path / "work")


def test_gate_requires_complete_review_matching_settings_and_evidence(tmp_path):
    with pytest.raises(runner.RunnerError, match="pending"):
        runner._check_gate(None, "fingerprint")
    evidence = tmp_path / "events.jsonl"
    evidence.write_text(events())
    gate = {key: True for key in ("execution_unavailable_or_explicitly_denied", "schema_transport_valid",
        "instructions_reviewed", "effective_declarations_reviewed", "managed_configuration_reviewed",
        "no_unverified_capabilities", "actual_exec_route_reviewed")}
    gate.update(fingerprint="fingerprint", reviewer="Fixture auditor", reviewed_at="2026-11-03",
                evidence=[{"path": evidence.name, "sha256": runner.sha256_file(evidence)}])
    path = tmp_path / "gate.json"
    runner.write_json(path, gate)
    assert runner._check_gate(path, "fingerprint") == gate
    with pytest.raises(runner.RunnerError, match="does not match"):
        runner._check_gate(path, "changed")
    evidence.write_text("changed")
    with pytest.raises(runner.RunnerError, match="changed"):
        runner._check_gate(path, "fingerprint")


def test_explicit_generation_stops_before_request_without_gate(repository, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_offline_inspect", lambda *a: "codex-cli fixture")
    monkeypatch.setattr(runner, "_execute", lambda *a: pytest.fail("Unvalidated game request"))
    with pytest.raises(runner.RunnerError, match="pending"):
        runner.prepare_run(repository, tmp_path / "run", tmp_path / "home", tmp_path / "work", execute=True)


@pytest.mark.parametrize("event_stream", [events(), (AUTHOR_FIXTURE / "events.jsonl").read_text()])
def test_explicit_preflight_keeps_runtime_denial_pending(repository, tmp_path, monkeypatch, event_stream):
    calls = []
    monkeypatch.setattr(runner, "_offline_inspect", lambda *a: "codex-cli fixture")
    def fixture_execution(argv, prompt, dest, env, work, timeout):
        calls.append(prompt.read_text())
        (dest / "events.jsonl").write_text(event_stream)
        (dest / "stderr.txt").write_text("")
        response = runner.TRANSPORT_FIXTURE if "structured-output" in prompt.read_text() else "Execution unavailable."
        (dest / "response.json").write_text(response)
        return {"execution": {"status": "finished", "exit_status": 0}, "audit": runner.audit_events(event_stream)}
    monkeypatch.setattr(runner, "_execute", fixture_execution)
    result = runner.prepare_run(repository, tmp_path / "preflight", tmp_path / "home", tmp_path / "work", mode="preflight", execute=True)
    gate = json.loads((tmp_path / "preflight/gate-review-template.json").read_text())
    assert len(calls) == 2 and result["smoke_requests"] == 2
    assert result["experimental_requests"] == 0
    assert gate["schema_transport_valid"]
    assert gate["execution_unavailable_or_explicitly_denied"] is False
    assert gate["no_unverified_capabilities"] is False
    assert result["runtime_preflight"] == "requires_manual_review"


def test_freeze_commits_hidden_disjoint_seeds_and_detects_changes(repository, tmp_path):
    calibration = tmp_path / "calibration.json"
    runner.write_json(calibration, {"passed": True, "seeds": list(range(256)), "trivial_comparator": "chase", "random_action_seed0": 10_000_019})
    frozen = tmp_path / "frozen"
    manifest = freeze_project(repository, frozen, calibration)
    test_manifest = json.loads((frozen / "test-seeds.private.json").read_text())
    seeds = test_manifest["seeds"]
    assert len(seeds) == len(set(seeds)) == 256
    assert not set(seeds).intersection(range(256))
    assert test_manifest["random_action_seed0"] == 20_000_019
    assert manifest["runtime_preflight"] == "pending" and not manifest["ready_for_generation"]
    assert json.loads((frozen / "readiness.json").read_text())["astra_generation"] == "not requested"
    assert json.loads((frozen / "diagnostic-probes.json").read_text())["reachable_single_object_count"] == 2496
    assert verify_freeze(repository, frozen / "manifest.json") == manifest
    assert "pyproject.toml" in manifest["files"] and "uv.lock" in manifest["files"]
    with pytest.raises(runner.RunnerError, match="already exists"):
        freeze_project(repository, frozen, calibration)
    (repository / "configs/game.json").write_text("changed")
    with pytest.raises(runner.RunnerError, match="changed"):
        verify_freeze(repository, frozen / "manifest.json")


def test_official_validation_rejects_a_dry_run(repository, tmp_path):
    frozen = tmp_path / "frozen"
    freeze_project(repository, frozen)
    runner.prepare_run(repository, tmp_path / "run", tmp_path / "home", tmp_path / "work")
    with pytest.raises(runner.RunnerError, match="explicitly executed"):
        runner.validate_generation_run(repository, tmp_path / "run", frozen / "manifest.json")
