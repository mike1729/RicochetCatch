"""Frozen, state-only development comparators. Exclude this file from model prompts.

Controllers receive only an immutable PolicyState. No tape, environment handle,
object identity, or evaluator trajectory labels are part of that interface.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from .env import PolicyState, clip_column, move_horizontal

ACTION_DELTAS = (0, -1, 1)
BASELINE_NAMES = (
    "stay", "random", "chase", "chase_safe", "one_step_lead_safe",
    "predict_greedy_safe", "reference",
)
TRIVIAL_NAMES = ("stay", "random", "chase")
RANDOM_ACTION_SEED0 = 10_000_019


def stay(state: PolicyState) -> int:
    return 0


class RandomPolicy:
    """Uniform actions from an RNG independent of scene generation."""

    def __init__(self, seed: int):
        self.seed = seed
        self._rng = random.Random(seed)

    def __call__(self, state: PolicyState) -> int:
        return self._rng.randrange(3)


def _lowest_gem(state: PolicyState):
    gems = [obj for obj in state.objects if obj.kind == "gem"]
    return max(gems, key=lambda obj: obj.row) if gems else None


def _toward(paddle: int, column: int) -> int:
    return 0 if paddle == column else (1 if column < paddle else 2)


def safe_actions(state: PolicyState) -> tuple[int, ...]:
    """Veto only next-tick bomb collisions using actual reflection mechanics."""
    unsafe = {
        move_horizontal(obj.column, obj.vx)[0]
        for obj in state.objects if obj.kind == "bomb" and obj.row == 6
    }
    safe = tuple(
        action for action, delta in enumerate(ACTION_DELTAS)
        if clip_column(state.paddle_column + delta) not in unsafe
    )
    assert safe, "A reachable state must admit an immediate safe action"
    return safe


def _safe_toward(state: PolicyState, target: int | None) -> int:
    safe = safe_actions(state)
    if target is None:
        return safe[0]
    return min(safe, key=lambda action: (
        abs(clip_column(state.paddle_column + ACTION_DELTAS[action]) - target), action,
    ))


def chase(state: PolicyState) -> int:
    gem = _lowest_gem(state)
    return 0 if gem is None else _toward(state.paddle_column, gem.column)


def chase_safe(state: PolicyState) -> int:
    gem = _lowest_gem(state)
    return _safe_toward(state, None if gem is None else gem.column)


def one_step_lead_safe(state: PolicyState) -> int:
    """One clipped step of gem lead, without forecasting eventual landings."""
    gem = _lowest_gem(state)
    target = None if gem is None else clip_column(gem.column + gem.vx)
    return _safe_toward(state, target)


def landing(row: int, column: int, vx: int) -> tuple[int, int]:
    ticks = 7 - row
    for _ in range(ticks):
        column, vx = move_horizontal(column, vx)
    return ticks, column


def predict_greedy_safe(state: PolicyState) -> int:
    candidates = []
    for obj in state.objects:
        if obj.kind == "gem":
            ticks, column = landing(obj.row, obj.column, obj.vx)
            distance = abs(column - state.paddle_column)
            if distance <= ticks:
                candidates.append((ticks, distance, column))
    return _safe_toward(state, min(candidates)[2] if candidates else None)


def reference(state: PolicyState) -> int:
    """Seven-tick visible-object DP with forbidden bomb cells and zero terminal value."""
    horizon = min(7, state.remaining_ticks)
    if horizon <= 0:
        raise ValueError("Cannot choose an action after the horizon")
    gems = [[0] * 8 for _ in range(horizon + 1)]
    bombs = [[False] * 8 for _ in range(horizon + 1)]
    for obj in state.objects:
        ticks, column = landing(obj.row, obj.column, obj.vx)
        if ticks <= horizon:
            if obj.kind == "gem":
                gems[ticks][column] += 1
            else:
                bombs[ticks][column] = True
    negative = -10**9
    values = [0] * 8
    for tick in range(horizon, 1, -1):
        following = []
        for paddle in range(8):
            options = [
                gems[tick][column] + values[column]
                for delta in ACTION_DELTAS
                for column in [clip_column(paddle + delta)]
                if not bombs[tick][column]
            ]
            following.append(max(options, default=negative))
        values = following
    safe = safe_actions(state)
    return max(safe, key=lambda action: (
        gems[1][clip_column(state.paddle_column + ACTION_DELTAS[action])]
        + values[clip_column(state.paddle_column + ACTION_DELTAS[action])], -action,
    ))


def make_baseline(name: str, episode_index: int = 0,
                  random_seed0: int = RANDOM_ACTION_SEED0) -> Callable[[PolicyState], int]:
    if name == "random":
        return RandomPolicy(random_seed0 + episode_index)
    policies = {
        "stay": stay, "chase": chase, "chase_safe": chase_safe,
        "one_step_lead_safe": one_step_lead_safe,
        "predict_greedy_safe": predict_greedy_safe, "reference": reference,
    }
    try:
        return policies[name]
    except KeyError:
        raise ValueError(f"Unknown baseline: {name}") from None
