from ricochet_catch.baselines import reference
from ricochet_catch.env import Object
from ricochet_catch.probes import (
    conflict_probe, construct_probe_state, generate_probes, probe_manifest,
    run_probe, run_probes,
)


def test_every_probe_is_distinct_and_has_a_legal_eight_tick_witness():
    probes = generate_probes()
    assert len(probes) == 2496
    assert len({probe.state for probe in probes}) == 2496
    for probe in probes:
        assert construct_probe_state(probe) == probe.state
        assert len(probe.prefix_actions) == 8
        assert probe.state.remaining_ticks == 248
        assert len(probe.state.objects) == 1
        assert len(probe.state.objects[0]) == 4


def test_unreachable_direction_states_excluded_after_first_move():
    states = {probe.state.objects[0] for probe in generate_probes()}
    assert Object("gem", 0, 0, 1) in states  # legal birth before any movement
    assert Object("gem", 1, 0, 1) not in states
    assert Object("bomb", 6, 7, -1) not in states
    assert Object("gem", 1, 0, -1) in states


def test_manifest_is_frozen_and_labels_are_explicit():
    manifest = probe_manifest()
    assert manifest["probe_set_sha256"] == "eee4fd50d10d82c91a248a59f0666e90bcacd9aaf9ecbf05d64ee70d67aaaee5"
    assert manifest["reachable_single_object_count"] == 2496
    assert manifest["reporting_rules"]["reflection"].startswith("a wall reflection strictly after")
    assert manifest["synthetic_validation_count"] == 0


def test_future_reflection_labels_and_catchability():
    probes = generate_probes()
    chosen = next(probe for probe in probes if probe.state.objects == (Object("gem", 4, 6, 1),)
                  and probe.state.paddle_column == 3)
    assert chosen.reflection_before_arrival
    assert chosen.ticks_to_arrival == 3 and chosen.landing_column == 5
    assert chosen.gem_catch_physically_reachable
    result = run_probe(reference, chosen)
    assert result["gem_caught"]
    assert len(result["trajectory"]) == 4


def test_conflict_is_reachable_and_reported_separately():
    probe = conflict_probe()
    assert construct_probe_state(probe) == probe.state
    result = run_probe(reference, probe)
    assert result["bomb_avoided"] and not result["gem_caught"]
    assert result["diagnostic_kind"] == "reachable_two_object_conflict"


def test_reference_catches_all_physically_reachable_gems_and_avoids_all_bombs():
    report = run_probes(lambda index: reference, include_trajectories=False)
    assert report["reachable_gem_catch"]["fraction"] == 1
    assert report["bomb_avoidance"]["states"] == 1248
    assert report["bomb_avoidance"]["fraction"] == 1
    assert len(report["single_object_results"]) == 2496
    assert "trajectory" not in report["conflict_result"]
    for label in ("velocity", "ticks_to_arrival", "reflection_before_arrival"):
        assert sum(group["bomb_avoidance"]["states"] for group in report["strata"][label].values()) == 1248
