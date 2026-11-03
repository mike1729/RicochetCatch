"""Frozen evaluation-only reachable probes. Never include this file in a prompt.

Every probe carries a legal eight-action construction from the ordinary reset.
Future-reflection and catchability labels belong to this evaluator, not state.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .env import Env, Object, PolicyState, Spawn, move_horizontal
from .evaluate import choose_action, state_record

PROBE_VERSION = "reachable-single-object-v1"
PROBE_ELAPSED_TICK = 8
EMPTY_TAPE = (None,) * 256


@dataclasses.dataclass(frozen=True)
class Probe:
    probe_id: str
    state: PolicyState
    prefix_actions: tuple[int, ...]
    prefix_spawns: tuple[tuple[int, Spawn], ...]
    ticks_to_arrival: int
    landing_column: int
    reflection_before_arrival: bool
    gem_catch_physically_reachable: bool
    diagnostic_kind: str = "reachable_single_object"

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id, "state": state_record(self.state),
            "prefix_actions": list(self.prefix_actions),
            "prefix_spawns": [{"tick": tick, "spawn": dict(spawn._asdict())}
                              for tick, spawn in self.prefix_spawns],
            "ticks_to_arrival": self.ticks_to_arrival, "landing_column": self.landing_column,
            "reflection_before_arrival": self.reflection_before_arrival,
            "gem_catch_physically_reachable": self.gem_catch_physically_reachable,
            "diagnostic_kind": self.diagnostic_kind,
        }


def _paddle_prefix(column: int) -> tuple[int, ...]:
    action = 1 if column < 3 else 2
    return (action,) * abs(column - 3) + (0,) * (8 - abs(column - 3))


def _future(obj: Object) -> tuple[int, int, bool]:
    column, vx, reflected = obj.column, obj.vx, False
    ticks = 7 - obj.row
    for _ in range(ticks):
        column, following = move_horizontal(column, vx)
        reflected |= following != vx
        vx = following
    return ticks, column, reflected


def construct_probe_state(probe: Probe) -> PolicyState:
    """Reconstruct the recorded witness using only legal reset transitions."""
    tape = list(EMPTY_TAPE)
    for tick, spawn in probe.prefix_spawns:
        tape[tick - 1] = spawn
    env = Env(tuple(tape))
    for action in probe.prefix_actions:
        env.step(action)
    if env.state != probe.state:
        raise ValueError(f"Invalid legal-prefix witness for {probe.probe_id}")
    return env.state


def generate_probes() -> tuple[Probe, ...]:
    """Enumerate distinct actual horizontal states after all legal birth prefixes."""
    probes = []
    seen = set()
    for kind in ("gem", "bomb"):
        for row in range(7):
            birth_tick = PROBE_ELAPSED_TICK - row
            for spawn_column in range(8):
                for spawn_vx in (-1, 0, 1):
                    spawn = Spawn(kind, spawn_column, spawn_vx)
                    column, vx = spawn_column, spawn_vx
                    for _ in range(row):
                        column, vx = move_horizontal(column, vx)
                    obj = Object(kind, row, column, vx)
                    for paddle in range(8):
                        key = (obj, paddle)
                        if key in seen:
                            continue
                        seen.add(key)
                        ticks, landing, reflected = _future(obj)
                        state = PolicyState(paddle, (obj,), 248)
                        probe = Probe(
                            f"single-{kind}-r{row}-c{column}-v{vx:+d}-p{paddle}", state,
                            _paddle_prefix(paddle), ((birth_tick, spawn),), ticks, landing,
                            reflected, kind == "gem" and abs(landing - paddle) <= ticks,
                        )
                        construct_probe_state(probe)
                        probes.append(probe)
    return tuple(probes)


def conflict_probe() -> Probe:
    probe = Probe("conflict-bomb-before-gem", PolicyState(3, (
        Object("bomb", 6, 4, 0), Object("gem", 5, 5, 0)), 248), (0,) * 8,
        ((2, Spawn("bomb", 4, 0)), (3, Spawn("gem", 5, 0))),
        2, 5, False, False, "reachable_two_object_conflict")
    construct_probe_state(probe)
    return probe


def probe_manifest() -> dict[str, Any]:
    records = [probe.to_dict() for probe in generate_probes()] + [conflict_probe().to_dict()]
    serialized = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "version": PROBE_VERSION, "elapsed_tick": PROBE_ELAPSED_TICK,
        "reachable_single_object_count": len(records) - 1,
        "reachable_conflict_count": 1, "synthetic_validation_count": 0,
        "probe_set_sha256": hashlib.sha256(serialized).hexdigest(),
        "reporting_rules": {
            "gem_denominator": "single-object gem states with abs(landing-paddle)<=ticks_to_arrival",
            "bomb_denominator": "all reachable single-object bomb states",
            "velocity": "observed horizontal velocity at elapsed tick 8",
            "reflection": "a wall reflection strictly after the probe state and no later than arrival",
            "stratification": ["velocity", "ticks_to_arrival", "reflection_before_arrival"],
            "conflict": "separate result, excluded from single-object rates",
            "synthetic": "excluded from reachable rates and gameplay evaluation",
        },
        "probes": records,
    }


def run_probe(policy, probe: Probe) -> dict[str, Any]:
    env = Env(EMPTY_TAPE, initial_state=probe.state, spawning=False)
    trajectory = [{"tick": 8, "state": state_record(env.state), "score": 0,
                   "action": None, "logits": None}]
    bomb_hit = False
    for _ in range(probe.ticks_to_arrival):
        if env.done:
            break
        action, logits, _ = choose_action(policy, env.state, env.observe())
        step = env.step(action)
        bomb_hit |= step.terminated
        trajectory.append({"tick": env.elapsed_ticks, "state": state_record(env.state),
                           "score": env.score, "action": action, "logits": logits,
                           "reward": step.reward, "terminated": step.terminated,
                           "truncated": step.truncated})
    first = probe.state.objects[0]
    return {"probe_id": probe.probe_id, "diagnostic_kind": probe.diagnostic_kind,
            "object_kind": first.kind if len(probe.state.objects) == 1 else "conflict",
            "velocity": first.vx if len(probe.state.objects) == 1 else None,
            "ticks_to_arrival": probe.ticks_to_arrival,
            "reflection_before_arrival": probe.reflection_before_arrival,
            "gem_catch_physically_reachable": probe.gem_catch_physically_reachable,
            "gem_caught": env.score == 1, "bomb_avoided": not bomb_hit,
            "trajectory": trajectory}


def _rate(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    successes = sum(record[key] for record in records)
    return {"successes": successes, "states": len(records),
            "fraction": successes / len(records) if records else None}


def run_probes(policy_factory, *, include_trajectories: bool = True) -> dict[str, Any]:
    """Use a fresh policy per controlled state; never alter submitted weights."""
    records = [run_probe(policy_factory(index), probe) for index, probe in enumerate(generate_probes())]
    conflict = run_probe(policy_factory(len(records)), conflict_probe())
    gems = [record for record in records if record["object_kind"] == "gem" and record["gem_catch_physically_reachable"]]
    bombs = [record for record in records if record["object_kind"] == "bomb"]
    strata = {}
    for field in ("velocity", "ticks_to_arrival", "reflection_before_arrival"):
        values = sorted({record[field] for record in records})
        strata[field] = {str(value): {
            "reachable_gem_catch": _rate([record for record in gems if record[field] == value], "gem_caught"),
            "bomb_avoidance": _rate([record for record in bombs if record[field] == value], "bomb_avoided"),
        } for value in values}
    if not include_trajectories:
        for record in records + [conflict]:
            record.pop("trajectory")
    return {
        "version": PROBE_VERSION, "probe_set_sha256": probe_manifest()["probe_set_sha256"],
        "reachable_gem_catch": _rate(gems, "gem_caught"), "bomb_avoidance": _rate(bombs, "bomb_avoided"),
        "strata": strata, "single_object_results": records, "conflict_result": conflict,
        "interpretation": "Controlled-state behavior does not identify an internal algorithm and cannot rescue the primary score.",
    }
