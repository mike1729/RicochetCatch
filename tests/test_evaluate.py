import dataclasses
import json

import numpy as np
import pytest

from ricochet_catch.baselines import stay
from ricochet_catch.env import Spawn, make_tape
from ricochet_catch.evaluate import (
    EpisodeResult, bootstrap_interval, bootstrap_lower, build_report, calibrate,
    eligible_gems, gem_stratum, report_summary, run_episode, save_episode,
    select_replays, verify_replay,
)


def fake(index, score, terminated=False):
    return EpisodeResult(index, "manual fixture", score, 10, terminated, not terminated,
                         {"vertical": 100, "diagonal_no_reflection": 0, "reflected": 0},
                         {"vertical": score, "diagonal_no_reflection": 0, "reflected": 0},
                         score, 0, 0)


def test_full_tape_denominators_include_gems_after_death_and_exclude_late_births():
    tape = [None] * 256
    tape[0] = Spawn("bomb", 3, 0)
    tape[1] = Spawn("gem", 3, 0)
    tape[248] = Spawn("gem", 3, 0)
    tape[249] = Spawn("gem", 3, 0)
    result = run_episode(stay, tuple(tape))
    assert result.length == 8 and result.terminated
    assert result.eligible["vertical"] == 2 and result.score == 0
    summary = report_summary([result])
    assert summary["missed_gems"] == 2
    assert summary["gem_opportunities"]["vertical"]["fraction"] == 0
    assert summary["survival_conditional"]["zero_denominator_episodes"] == 1


def test_full_fall_reflection_labels():
    assert gem_stratum(Spawn("gem", 0, 0)) == "vertical"
    assert gem_stratum(Spawn("gem", 0, 1)) == "diagonal_no_reflection"
    assert gem_stratum(Spawn("gem", 7, 1)) == "reflected"
    tape = [None] * 256
    tape[0] = Spawn("gem", 3, 0)
    tape[248] = Spawn("gem", 3, 0)
    result = run_episode(stay, tuple(tape))
    assert result.score == 2 and result.caught["vertical"] == 2
    assert result.truncated and result.length == 256


def test_diagnostic_labels_cannot_change_actions_or_observations(monkeypatch):
    from ricochet_catch import evaluate
    from ricochet_catch.baselines import reference
    tape = make_tape(12)
    first = run_episode(reference, tape)
    monkeypatch.setattr(evaluate, "gem_stratum", lambda spawn: "vertical")
    second = run_episode(reference, tape)
    assert first.trajectory == second.trajectory


def test_logit_ties_recorded_and_nonfinite_values_rejected():
    class Zero:
        def logits(self, observation):
            assert observation.shape == (449,)
            return np.zeros(3, dtype=np.float32)
    result = run_episode(Zero(), (None,) * 256)
    assert result.near_tie_decisions == 256
    assert all(frame["action"] == 0 for frame in result.trajectory[1:])
    class Bad:
        def logits(self, observation):
            return [0, np.nan, 1]
    with pytest.raises(ValueError, match="finite"):
        run_episode(Bad(), (None,) * 256)


def test_bootstrap_precommitted_rank_and_constant_data():
    assert bootstrap_lower([2.0] * 8, resamples=37) == 2
    assert bootstrap_interval([2.0] * 8, resamples=37) == (2, 2)
    assert bootstrap_lower([-1, 1], seed=123, resamples=100) == -1
    assert bootstrap_interval([0, 1, 2], seed=3, resamples=100) == bootstrap_interval([0, 1, 2], seed=3, resamples=100)


def test_fixed_replay_selection_uses_indices_to_break_ties():
    results = [fake(index, index // 2, terminated=index in (3, 7)) for index in range(256)]
    assert select_replays(results) == {"lower_median": 127, "highest_score": 254, "earliest_bomb_failure": 3}
    assert select_replays([fake(9, 0), fake(2, 0)]) == {
        "lower_median": 2, "highest_score": 2, "earliest_bomb_failure": None}


def test_success_requires_both_paired_bounds_parse_audit_and_reference():
    network = [fake(index, 8) for index in range(8)]
    baselines = {"stay": [fake(index, 0) for index in range(8)],
                 "reference": [fake(index, 10) for index in range(8)]}
    report = build_report(network, baselines, "stay", parse_valid=True, audit_valid=True,
                          expected_episodes=8, resamples=100)
    assert report["successful_demonstration"]
    assert report["paired_comparisons"]["halfway_margin"]["lower_one_sided_95"] == 3
    for parse, audit in ((False, True), (True, False)):
        assert not build_report(network, baselines, "stay", parse_valid=parse, audit_valid=audit,
                                expected_episodes=8, resamples=5)["successful_demonstration"]
    baselines["reference"] = baselines["stay"]
    assert not build_report(network, baselines, "stay", parse_valid=True, audit_valid=True,
                            expected_episodes=8, resamples=5)["successful_demonstration"]


def test_reports_reject_unpaired_or_incomplete_episodes():
    with pytest.raises(ValueError, match="complete"):
        build_report([fake(0, 4)], {"stay": [fake(0, 0)], "reference": [fake(0, 5)]},
                     "stay", parse_valid=True, audit_valid=True)
    with pytest.raises(ValueError, match="paired"):
        build_report([fake(0, 4)], {"stay": [fake(1, 0)], "reference": [fake(0, 5)]},
                     "stay", parse_valid=True, audit_valid=True, expected_episodes=1)
    with pytest.raises(ValueError, match="Spawn tape"):
        build_report([dataclasses.replace(fake(0, 4), tape_sha256="scene-a")],
                     {"stay": [dataclasses.replace(fake(0, 0), tape_sha256="scene-b")],
                      "reference": [dataclasses.replace(fake(0, 5), tape_sha256="scene-a")]},
                     "stay", parse_valid=True, audit_valid=True, expected_episodes=1)


def test_round_trip_replay_and_partial_recording(tmp_path):
    result = run_episode(stay, make_tape(9))
    path = tmp_path / "replay.json"
    save_episode(result, path, selection="highest_score")
    record = json.loads(path.read_text())
    verify_replay(record)
    record["trajectory"] = record["trajectory"][:5]
    record["score"] = record["trajectory"][-1]["score"]
    verify_replay(record)


@pytest.mark.parametrize("field", ["score", "tick", "action", "reward"])
def test_replay_tampering_rejected(tmp_path, field):
    result = run_episode(stay, (None,) * 256)
    path = tmp_path / "replay.json"
    save_episode(result, path, selection="synthetic test")
    record = json.loads(path.read_text())
    record["trajectory"][1][field] += 1
    with pytest.raises(ValueError):
        verify_replay(record)


def test_frames_after_completion_rejected(tmp_path):
    result = run_episode(stay, (None,) * 256)
    path = tmp_path / "replay.json"
    save_episode(result, path, selection="synthetic test")
    record = json.loads(path.read_text())
    record["trajectory"].append(record["trajectory"][-1])
    with pytest.raises(ValueError, match="completion"):
        verify_replay(record)
