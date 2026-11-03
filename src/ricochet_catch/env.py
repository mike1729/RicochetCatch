"""Standalone game physics. RNG: CPython random.Random (MT19937), four draws/tick.

Only PolicyState and its observation may be supplied to policies. The evaluator
owns tapes; no seed, randomness, future arrival, or diagnostic label is in state.
"""
from dataclasses import dataclass
import math
import random
from typing import NamedTuple

WIDTH = HEIGHT = 8
PADDLE_ROW = 7
HORIZON = 256
GEM, BOMB = "gem", "bomb"
ACTION_DELTAS = (0, -1, 1)


def _integer(value, lower, upper, label):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{label} must be an integer in [{lower}, {upper}]")


@dataclass(frozen=True)
class GameConfig:
    spawn_probability: float = 0.45
    bomb_probability: float = 0.25
    velocity_probabilities: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    horizon: int = HORIZON

    def __post_init__(self):
        if type(self.horizon) is not int or self.horizon != HORIZON:
            raise ValueError("game v0.1 requires horizon=256")
        for value in (self.spawn_probability, self.bomb_probability):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("probabilities must be finite and in [0,1]")
        probs = tuple(self.velocity_probabilities)
        if len(probs) != 3 or any(isinstance(p, bool) or not math.isfinite(p) or p < 0 for p in probs):
            raise ValueError("three finite nonnegative velocity probabilities required")
        if not math.isclose(sum(probs), 1.0, abs_tol=1e-12):
            raise ValueError("velocity probabilities must sum to one")
        object.__setattr__(self, "velocity_probabilities", probs)


class Object(NamedTuple):
    kind: str
    row: int
    column: int
    vx: int


class Spawn(NamedTuple):
    kind: str
    column: int
    vx: int


class PolicyState(NamedTuple):
    paddle_column: int
    objects: tuple[Object, ...]
    remaining_ticks: int


class Event(NamedTuple):
    kind: str
    object: Object | None = None
    paddle_column: int | None = None


class StepResult(NamedTuple):
    observation: object
    reward: int
    terminated: bool
    truncated: bool
    events: tuple[Event, ...]


def clip_column(column):
    return max(0, min(WIDTH - 1, column))


def move_horizontal(column, vx):
    next_column = column + vx
    if not 0 <= next_column < WIDTH:
        vx = -vx
        next_column = column + vx
    return next_column, vx


def validate_state(state):
    if not isinstance(state, PolicyState):
        raise ValueError("expected a PolicyState")
    _integer(state.paddle_column, 0, 7, "paddle_column")
    _integer(state.remaining_ticks, 0, HORIZON, "remaining_ticks")
    if not isinstance(state.objects, tuple):
        raise ValueError("objects must be an immutable tuple")
    rows = set()
    for obj in state.objects:
        if not isinstance(obj, Object) or obj.kind not in (GEM, BOMB):
            raise ValueError("invalid object")
        _integer(obj.row, 0, 6, "row")
        _integer(obj.column, 0, 7, "column")
        _integer(obj.vx, -1, 1, "vx")
        if obj.row in rows:
            raise ValueError("objects must occupy unique rows")
        rows.add(obj.row)


def make_tape(seed, config=GameConfig()):
    """Immutable 256-tick tape; draw spawn,column,velocity,type even if absent."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    entries = []
    p0, p1, _ = config.velocity_probabilities
    for _ in range(config.horizon):
        u_spawn, u_col, u_vel, u_type = (rng.random() for _ in range(4))
        column = min(7, int(u_col * WIDTH))
        vx = -1 if u_vel < p0 else (0 if u_vel < p0 + p1 else 1)
        kind = BOMB if u_type < config.bomb_probability else GEM
        entries.append(Spawn(kind, column, vx) if u_spawn < config.spawn_probability else None)
    return tuple(entries)


class Env:
    def __init__(self, tape, config=GameConfig(), initial_state=None, spawning=True):
        self.config = config
        self._tape = tuple(tape)
        if len(self._tape) != config.horizon:
            raise ValueError("tape must contain exactly 256 entries")
        for entry in self._tape:
            if entry is not None:
                if not isinstance(entry, Spawn) or entry.kind not in (GEM, BOMB):
                    raise ValueError("invalid spawn entry")
                _integer(entry.column, 0, 7, "spawn column")
                _integer(entry.vx, -1, 1, "spawn vx")
        self._state = PolicyState(3, (), config.horizon) if initial_state is None else initial_state
        validate_state(self._state)
        self.elapsed_ticks = config.horizon - self._state.remaining_ticks
        self.score = 0
        self.terminated = False
        self.truncated = self._state.remaining_ticks == 0
        self.spawning = bool(spawning)

    @property
    def state(self):
        return self._state

    @property
    def done(self):
        return self.terminated or self.truncated

    def observe(self):
        from .observations import observe
        return observe(self._state)

    def step(self, action):
        if self.done:
            raise RuntimeError("step() on finished episode")
        _integer(action, 0, 2, "action")
        paddle = clip_column(self._state.paddle_column + ACTION_DELTAS[action])
        survivors, events = [], []
        reward, hit = 0, False
        for obj in self._state.objects:
            column, vx = move_horizontal(obj.column, obj.vx)
            moved = Object(obj.kind, obj.row + 1, column, vx)
            if moved.row < PADDLE_ROW:
                survivors.append(moved)
            else:
                caught = column == paddle
                if moved.kind == GEM:
                    reward += int(caught)
                    event = "gem_caught" if caught else "gem_missed"
                else:
                    hit |= caught
                    event = "bomb_hit" if caught else "bomb_missed"
                events.append(Event(event, moved, paddle))
        self.elapsed_ticks += 1
        remaining = self._state.remaining_ticks - 1
        self.score += reward
        self.terminated = hit
        self.truncated = not hit and remaining == 0
        if not self.done and self.spawning:
            entry = self._tape[self.elapsed_ticks - 1]
            if entry is not None:
                spawned = Object(entry.kind, 0, entry.column, entry.vx)
                survivors.append(spawned)
                events.append(Event("spawn", spawned, paddle))
        self._state = PolicyState(paddle, tuple(survivors), remaining)
        return StepResult(self.observe(), reward, self.terminated, self.truncated, tuple(events))
