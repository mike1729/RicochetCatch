"""Development calibration, paired reports, and deterministic recorded playback.

This evaluator never runs a model request. Held-out tapes are evaluated only by
an explicit caller; calibration uses development seeds 0 through 255.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import random
import statistics
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import BASELINE_NAMES, RANDOM_ACTION_SEED0, TRIVIAL_NAMES, make_baseline
from .env import Env, GameConfig, Object, PolicyState, Spawn, make_tape, move_horizontal, validate_state

BOOTSTRAP_SEED = 12345
BOOTSTRAP_RESAMPLES = 10_000
NEAR_TIE_TOLERANCE = 1e-6
GEM_STRATA = ("vertical", "diagonal_no_reflection", "reflected")


def state_record(state: PolicyState) -> dict[str, Any]:
    return {
        "paddle_column": state.paddle_column,
        "objects": [dict(obj._asdict()) for obj in state.objects],
        "remaining_ticks": state.remaining_ticks,
    }


def _event_record(event) -> dict[str, Any]:
    return {"kind": event.kind,
            "object": dict(event.object._asdict()) if event.object is not None else None,
            "paddle_column": event.paddle_column}


def choose_action(policy, state: PolicyState, observation: np.ndarray) -> tuple[int, list[float] | None, bool]:
    """Pass exactly the observation or state; never pass an environment handle."""
    if hasattr(policy, "logits"):
        logits = np.asarray(policy.logits(observation), dtype=np.float32)
        if logits.shape != (3,) or not np.isfinite(logits).all():
            raise ValueError("Policy logits must be three finite float32 values")
        ordered = np.sort(logits.astype(np.float64))
        near_tie = ordered[-1] - ordered[-2] <= NEAR_TIE_TOLERANCE * max(1.0, float(np.max(np.abs(logits))))
        return int(np.argmax(logits)), logits.tolist(), bool(near_tie)
    return policy(state), None, False


def gem_stratum(spawn: Spawn) -> str:
    """Full seven-move fall label, calculated outside the policy interface."""
    if spawn.vx == 0:
        return "vertical"
    column, vx = spawn.column, spawn.vx
    reflected = False
    for _ in range(7):
        column, following_vx = move_horizontal(column, vx)
        reflected |= following_vx != vx
        vx = following_vx
    return "reflected" if reflected else "diagonal_no_reflection"


def eligible_gems(tape: Sequence[Spawn | None], horizon: int) -> dict[str, int]:
    counts = dict.fromkeys(GEM_STRATA, 0)
    for index, spawn in enumerate(tape):
        # Birth is after tick index+1; seven later object moves reach row 7.
        if index + 8 <= horizon and spawn is not None and spawn.kind == "gem":
            counts[gem_stratum(spawn)] += 1
    return counts


@dataclasses.dataclass(frozen=True)
class EpisodeResult:
    episode_index: int
    policy_name: str
    score: int
    length: int
    terminated: bool
    truncated: bool
    eligible: dict[str, int]
    caught: dict[str, int]
    landed_gems: int
    missed_before_termination: int
    near_tie_decisions: int
    trajectory: tuple[dict[str, Any], ...] = ()
    tape_sha256: str = ""

    @property
    def bomb_hit(self) -> bool:
        return self.terminated

    def to_dict(self, include_trajectory: bool = True) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        if not include_trajectory:
            result.pop("trajectory")
        return result


def run_episode(policy, tape: Sequence[Spawn | None], config: GameConfig | None = None,
                episode_index: int = 0, policy_name: str = "policy", record: bool = True) -> EpisodeResult:
    config = config or GameConfig()
    env = Env(tape, config)
    eligible = eligible_gems(tape, config.horizon)
    caught = dict.fromkeys(GEM_STRATA, 0)
    landed_gems = missed = near_ties = 0
    trajectory = []
    if record:
        trajectory.append({"tick": 0, "state": state_record(env.state), "score": 0,
                           "action": None, "logits": None, "events": [],
                           "reward": 0, "terminated": False, "truncated": False})
    while not env.done:
        action, logits, near_tie = choose_action(policy, env.state, env.observe())
        near_ties += int(near_tie)
        step = env.step(action)
        for event in step.events:
            if event.kind in ("gem_caught", "gem_missed"):
                landed_gems += 1
            if event.kind == "gem_missed":
                missed += 1
            elif event.kind == "gem_caught":
                spawn_index = env.elapsed_ticks - 8
                spawn = tape[spawn_index]
                assert spawn is not None and spawn.kind == "gem"
                caught[gem_stratum(spawn)] += 1
        if record:
            trajectory.append({"tick": env.elapsed_ticks, "state": state_record(env.state),
                               "score": env.score, "action": action, "logits": logits,
                               "events": [_event_record(event) for event in step.events],
                               "reward": step.reward, "terminated": step.terminated,
                               "truncated": step.truncated})
    tape_hash = hashlib.sha256(json.dumps(list(tape), separators=(",", ":")).encode()).hexdigest()
    return EpisodeResult(episode_index, policy_name, env.score, env.elapsed_ticks,
                         step.terminated, step.truncated, eligible, caught, landed_gems,
                         missed, near_ties, tuple(trajectory), tape_hash)


def evaluate_baselines(seeds: Iterable[int], config: GameConfig | None = None,
                       names: Sequence[str] = BASELINE_NAMES, record: bool = False,
                       random_seed0: int = RANDOM_ACTION_SEED0) -> dict[str, list[EpisodeResult]]:
    config = config or GameConfig()
    tapes = [make_tape(seed, config) for seed in seeds]
    return {name: [run_episode(make_baseline(name, index, random_seed0), tape, config,
                              index, name, record)
                   for index, tape in enumerate(tapes)] for name in names}


def evaluate_policy(policy_factory: Callable[[int], Any], seeds: Iterable[int],
                    config: GameConfig | None = None, policy_name: str = "network",
                    record: bool = True) -> list[EpisodeResult]:
    config = config or GameConfig()
    return [run_episode(policy_factory(index), make_tape(seed, config), config,
                        index, policy_name, record) for index, seed in enumerate(seeds)]


def bootstrap_means(values: Sequence[float], *, seed: int = BOOTSTRAP_SEED,
                    resamples: int = BOOTSTRAP_RESAMPLES) -> np.ndarray:
    if len(values) == 0 or type(resamples) is not int or resamples < 1:
        raise ValueError("Bootstrap requires nonempty values and positive resamples")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Bootstrap values must be a finite vector")
    rng = random.Random(seed)
    count = len(values)
    means = np.empty(resamples, dtype=np.float64)
    # Python's recorded independent RNG also reproduces the review simulation.
    for index in range(resamples):
        means[index] = sum(values[rng.randrange(count)] for _ in range(count)) / count
    means.sort()
    return means


def _percentile(ordered: np.ndarray, fraction: float) -> float:
    """Frozen empirical rank: floor(q*N), clamped at N-1, no interpolation."""
    return float(ordered[min(len(ordered) - 1, int(fraction * len(ordered)))])


def bootstrap_interval(values: Sequence[float], *, seed: int = BOOTSTRAP_SEED,
                       resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    ordered = bootstrap_means(values, seed=seed, resamples=resamples)
    return _percentile(ordered, .025), _percentile(ordered, .975)


def bootstrap_lower(values: Sequence[float], *, seed: int = BOOTSTRAP_SEED,
                    resamples: int = BOOTSTRAP_RESAMPLES) -> float:
    return _percentile(bootstrap_means(values, seed=seed, resamples=resamples), .05)


def report_summary(results: Sequence[EpisodeResult], *, bootstrap: bool = False,
                   bootstrap_seed: int = BOOTSTRAP_SEED,
                   resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot summarize an empty evaluation")
    scores = [result.score for result in results]
    eligible = {label: sum(result.eligible[label] for result in results) for label in GEM_STRATA}
    caught = {label: sum(result.caught[label] for result in results) for label in GEM_STRATA}
    landed = sum(result.landed_gems for result in results)
    summary = {
        "episodes": len(results), "mean": statistics.mean(scores),
        "median": statistics.median(scores), "standard_deviation": statistics.pstdev(scores),
        "standard_deviation_convention": "population",
        "catches": sum(scores), "missed_gems": sum(eligible.values()) - sum(scores),
        "bomb_collision_fraction": sum(result.terminated for result in results) / len(results),
        "mean_episode_length": statistics.mean(result.length for result in results),
        "gem_opportunities": {
            label: {"eligible_full_tape": eligible[label], "caught": caught[label],
                    "fraction": caught[label] / eligible[label] if eligible[label] else None}
            for label in GEM_STRATA
        },
        "survival_conditional": {
            "landed_gems": landed, "caught": sum(scores),
            "catch_fraction": sum(scores) / landed if landed else None,
            "zero_denominator_episodes": sum(result.landed_gems == 0 for result in results),
            "missed_before_termination": sum(result.missed_before_termination for result in results),
        },
        "near_tie_decisions": sum(result.near_tie_decisions for result in results),
    }
    if bootstrap:
        summary["mean_bootstrap_95_interval"] = bootstrap_interval(scores, seed=bootstrap_seed, resamples=resamples)
    return summary


def select_replays(results: Sequence[EpisodeResult]) -> dict[str, int | None]:
    if not results:
        raise ValueError("Cannot select from no episodes")
    ordered = sorted(results, key=lambda result: (result.score, result.episode_index))
    return {
        "lower_median": ordered[(len(ordered) - 1) // 2].episode_index,
        "highest_score": min(results, key=lambda result: (-result.score, result.episode_index)).episode_index,
        "earliest_bomb_failure": min((result.episode_index for result in results if result.terminated), default=None),
    }


def calibrate(config: GameConfig | None = None, seeds: Iterable[int] = range(256),
              *, bootstrap_seed: int = BOOTSTRAP_SEED,
              resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    config = config or GameConfig()
    seeds = list(seeds)
    results = evaluate_baselines(seeds, config)
    summary = {name: report_summary(episodes) for name, episodes in results.items()}
    comparator = max(TRIVIAL_NAMES, key=lambda name: (summary[name]["mean"], -TRIVIAL_NAMES.index(name)))
    reference = summary["reference"]["mean"]
    checks = {
        "reference_mean_at_least_10": reference >= 10,
        "reference_beats_safe_chase_by_20_percent": reference >= 1.2 * summary["chase_safe"]["mean"],
        "reference_beats_trivial_by_10": reference >= summary[comparator]["mean"] + 10,
        "nonzero_bombs_and_all_velocities_at_least_half_diagonal": (
            config.bomb_probability > 0 and all(probability > 0 for probability in config.velocity_probabilities)
            and config.velocity_probabilities[0] + config.velocity_probabilities[2] >= .5),
    }
    margins = [results["one_step_lead_safe"][index].score - .5 * (
        results["reference"][index].score + results[comparator][index].score)
        for index in range(len(seeds))]
    return {
        "kind": "development_calibration", "astra_result": False,
        "config": dataclasses.asdict(config), "seeds": seeds,
        "random_action_seed0": RANDOM_ACTION_SEED0, "random_action_rng": "Python random.Random MT19937",
        "bootstrap_seed": bootstrap_seed, "bootstrap_resamples": resamples,
        "bootstrap_percentile_rule": "sorted means at floor(q*N), clamped at N-1",
        "passed": all(checks.values()), "checks": checks, "trivial_comparator": comparator,
        "halfway_threshold": .5 * (reference + summary[comparator]["mean"]),
        "one_step_lead_halfway_margin_lower_95": bootstrap_lower(margins, seed=bootstrap_seed, resamples=resamples),
        "summary": summary,
        "episodes": {name: [result.to_dict(False) for result in episodes] for name, episodes in results.items()},
        "replay_selection": {name: select_replays(episodes) for name, episodes in results.items()},
        "calibration_changes": [
            {"constant": name, "original": original, "calibrated": dataclasses.asdict(config)[name]}
            for name, original in dataclasses.asdict(GameConfig()).items()
            if dataclasses.asdict(config)[name] != original
        ],
    }


def build_report(network_results: Sequence[EpisodeResult], baseline_results: dict[str, Sequence[EpisodeResult]],
                 trivial_comparator: str, *, parse_valid: bool, audit_valid: bool,
                 bootstrap_seed: int = BOOTSTRAP_SEED, resamples: int = BOOTSTRAP_RESAMPLES,
                 metadata: dict[str, Any] | None = None, expected_episodes: int = 256) -> dict[str, Any]:
    if trivial_comparator not in TRIVIAL_NAMES:
        raise ValueError("Comparator must be selected from the declared trivial baselines during development")
    required = {"reference", trivial_comparator}
    if not required <= set(baseline_results):
        raise ValueError("Missing frozen comparator episode results")
    indexed = [result.episode_index for result in network_results]
    if len(indexed) != expected_episodes or len(set(indexed)) != len(indexed):
        raise ValueError("Official report requires the complete unique episode set")
    for name, results in baseline_results.items():
        if [result.episode_index for result in results] != indexed:
            raise ValueError(f"Episode indices for {name} are not paired with network results")
        for network, baseline in zip(network_results, results):
            if network.tape_sha256 != baseline.tape_sha256:
                raise ValueError(f"Spawn tape for {name} differs from the paired network episode")
    reference = baseline_results["reference"]
    trivial = baseline_results[trivial_comparator]
    halfway = [network.score - .5 * (ref.score + tri.score)
               for network, ref, tri in zip(network_results, reference, trivial)]
    versus_trivial = [network.score - tri.score for network, tri in zip(network_results, trivial)]
    versus_reference = [network.score - ref.score for network, ref in zip(network_results, reference)]
    comparisons = {}
    for name, differences in (("halfway_margin", halfway), ("versus_trivial", versus_trivial),
                              ("versus_reference", versus_reference)):
        ordered = bootstrap_means(differences, seed=bootstrap_seed, resamples=resamples)
        comparisons[name] = {"mean": statistics.mean(differences),
                             "lower_one_sided_95": _percentile(ordered, .05),
                             "two_sided_95_interval": (_percentile(ordered, .025), _percentile(ordered, .975))}
    reference_valid = statistics.mean(result.score for result in reference) > statistics.mean(result.score for result in trivial)
    passed = bool(parse_valid and audit_valid and reference_valid
                  and comparisons["halfway_margin"]["lower_one_sided_95"] > 0
                  and comparisons["versus_trivial"]["lower_one_sided_95"] > 0)
    return {
        "kind": "frozen_policy_evaluation", "successful_demonstration": passed,
        "parse_valid": parse_valid, "tool_audit_valid": audit_valid,
        "reference_calibration_valid": reference_valid, "trivial_comparator": trivial_comparator,
        "bootstrap": {"seed": bootstrap_seed, "resamples": resamples,
                      "joint_episode_resampling": True,
                      "percentile_rule": "sorted means at floor(q*N), clamped at N-1"},
        "primary_arithmetic": "float32", "near_tie_tolerance": NEAR_TIE_TOLERANCE,
        "summary": {"network": report_summary(network_results, bootstrap=True, bootstrap_seed=bootstrap_seed, resamples=resamples),
                    **{name: report_summary(results, bootstrap=True, bootstrap_seed=bootstrap_seed, resamples=resamples)
                       for name, results in baseline_results.items()}},
        "paired_comparisons": comparisons, "selected_replays": select_replays(network_results),
        "metadata": metadata or {},
        "interpretation": "Episode scores describe the fixed submitted network; diagnostics do not identify its internal algorithm.",
    }


def save_episode(result: EpisodeResult, path: str | Path, *, selection: str) -> None:
    Path(path).write_text(json.dumps({"format": "ricochet-replay-v1", "selection": selection,
                                     **result.to_dict()}, indent=2) + "\n")


def verify_replay(record: dict[str, Any]) -> None:
    """Validate recorded physics without a policy or any random generation."""
    if record.get("format") != "ricochet-replay-v1":
        raise ValueError("Unknown replay format")
    frames = record.get("trajectory")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("Replay has no complete trajectory")
    first = frames[0]
    data = first["state"]
    validate_state(PolicyState(data["paddle_column"], tuple(Object(**obj) for obj in data["objects"]), data["remaining_ticks"]))
    if (type(first["tick"]) is not int or first["tick"] != 256 - data["remaining_ticks"]
            or type(first["score"]) is not int or first["score"] < 0):
        raise ValueError("Invalid initial tick or score")
    for previous, following in zip(frames, frames[1:]):
        if previous.get("terminated") or previous.get("truncated"):
            raise ValueError("Replay has a frame after episode completion")
        if (type(following["tick"]) is not int or following["tick"] != previous["tick"] + 1
                or following["tick"] != 256 - following["state"]["remaining_ticks"]
                or type(following["reward"]) is not int or type(following["score"]) is not int
                or following["score"] != previous["score"] + following["reward"]):
            raise ValueError("Replay tick or score accounting mismatch")
        data = previous["state"]
        state = PolicyState(data["paddle_column"], tuple(Object(**obj) for obj in data["objects"]), data["remaining_ticks"])
        # A one-tick replay tape contains only the already recorded spawn event.
        tape = [None] * 256
        for event in following["events"]:
            if event["kind"] == "spawn":
                obj = event["object"]
                tape[256 - state.remaining_ticks] = Spawn(obj["kind"], obj["column"], obj["vx"])
        env = Env(tuple(tape), initial_state=state)
        step = env.step(following["action"])
        if (state_record(env.state) != following["state"] or step.reward != following["reward"]
                or step.terminated != following["terminated"] or step.truncated != following["truncated"]
                or [_event_record(event) for event in step.events] != following["events"]):
            raise ValueError(f"Replay physics mismatch at tick {following['tick']}")
    if record.get("score", frames[-1]["score"]) != frames[-1]["score"]:
        raise ValueError("Replay final score differs from trajectory")


def runtime_metadata() -> dict[str, Any]:
    import os
    return {"python": platform.python_version(), "numpy": np.__version__,
            "cpu_architecture": platform.machine(), "platform": platform.platform(),
            "blas_config": np.show_config(mode="dicts"),
            "thread_settings": {name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}}
