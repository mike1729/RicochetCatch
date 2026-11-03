import importlib.util
from pathlib import Path
import random

import numpy as np
import pytest

from ricochet_catch.env import (
    BOMB, GEM, Env, GameConfig, Object, PolicyState, Spawn, make_tape,
)
from ricochet_catch.observations import grid_index, observe

EMPTY_TAPE = (None,) * 256


def fixture(objects=(), paddle=3, remaining=100):
    return Env(EMPTY_TAPE, initial_state=PolicyState(paddle, tuple(objects), remaining), spawning=False)


@pytest.mark.parametrize("kind,column,vx,expected", [
    (GEM, 7, 1, Object(GEM, 4, 6, -1)),
    (BOMB, 0, -1, Object(BOMB, 4, 1, 1)),
    (GEM, 6, 1, Object(GEM, 4, 7, 1)),
])
def test_wall_reflection(kind, column, vx, expected):
    env = fixture([Object(kind, 3, column, vx)])
    env.step(0)
    assert env.state.objects == (expected,)
    if column == 6:
        env.step(0)
        assert env.state.objects == (Object(GEM, 5, 6, -1),)


@pytest.mark.parametrize("kind,action,reward,terminated", [
    (GEM, 0, 1, False), (BOMB, 0, 0, True), (BOMB, 1, 0, False),
])
def test_new_paddle_cell_scoring_and_removal(kind, action, reward, terminated):
    env = fixture([Object(kind, 6, 3, 0)])
    result = env.step(action)
    assert (result.reward, result.terminated, result.truncated) == (reward, terminated, False)
    assert env.score == reward
    assert env.state.objects == ()
    assert env.state.remaining_ticks == 99


@pytest.mark.parametrize("kind,reward,terminated,truncated", [
    (GEM, 1, False, True), (BOMB, 0, True, False),
])
def test_last_tick_collision_precedence(kind, reward, terminated, truncated):
    env = fixture([Object(kind, 6, 3, 0)], remaining=1)
    result = env.step(0)
    assert (result.reward, result.terminated, result.truncated) == (reward, terminated, truncated)
    assert result.observation[448] == 0
    with pytest.raises(RuntimeError):
        env.step(0)


def test_spawn_after_movement_and_no_spawn_when_finished():
    tape = (Spawn(GEM, 2, 1),) * 256
    env = Env(tape)
    env.step(0)
    assert env.state.objects == (Object(GEM, 0, 2, 1),)
    env.step(0)
    assert env.state.objects == (Object(GEM, 1, 3, 1), Object(GEM, 0, 2, 1))
    for state in [PolicyState(3, (), 1), PolicyState(3, (Object(BOMB, 6, 3, 0),), 100)]:
        finished = Env(tape, initial_state=state)
        result = finished.step(0)
        assert not any(e.kind == "spawn" for e in result.events)


@pytest.mark.parametrize("paddle,action", [(0, 1), (7, 2)])
def test_wall_clipping(paddle, action):
    env = fixture(paddle=paddle)
    env.step(action)
    assert env.state.paddle_column == paddle


@pytest.mark.parametrize("bad", [-1, 3, True, False, 1.0, "1", None, np.int64(1)])
def test_invalid_action_does_not_change_state(bad):
    env = fixture()
    before = env.state
    with pytest.raises(ValueError):
        env.step(bad)
    assert env.state == before and env.score == 0


def test_prediction_fixture():
    for actions, score in [([2, 2, 0], 1), ([2, 2, 2], 0)]:
        env = fixture([Object(GEM, 4, 6, 1)])
        for action in actions:
            env.step(action)
        assert env.score == score


def test_conflicting_objectives_all_action_pairs():
    for a1 in range(3):
        for a2 in range(3):
            env = fixture([Object(BOMB, 6, 4, 0), Object(GEM, 5, 5, 0)])
            env.step(a1)
            if not env.done:
                env.step(a2)
                assert env.score == 0


@pytest.mark.parametrize("state", [
    (), False, {},
    PolicyState(-1, (), 100), PolicyState(True, (), 100),
    PolicyState(3, (), 257), PolicyState(3, (), -1),
    PolicyState(3, (Object(GEM, 7, 0, 1),), 100),
    PolicyState(3, (Object(GEM, 0, 8, 1),), 100),
    PolicyState(3, (Object("x", 0, 0, 1),), 100),
    PolicyState(3, (Object(GEM, 0, 0, 2),), 100),
    PolicyState(3, (Object(GEM, 0, 0, 1), Object(BOMB, 0, 1, 0)), 100),
])
def test_state_loader_rejects_unreachable_fields(state):
    with pytest.raises(ValueError):
        Env(EMPTY_TAPE, initial_state=state)


def test_exact_observation_indices_and_dtype():
    objects = (Object(GEM, 0, 2, -1), Object(GEM, 1, 3, 0),
               Object(GEM, 2, 4, 1), Object(BOMB, 3, 1, -1),
               Object(BOMB, 4, 5, 0), Object(BOMB, 5, 6, 1))
    x = observe(PolicyState(3, objects, 128))
    expected = {((7 * 8 + 3) * 7), 15, 79, 143, 179, 264, 328}
    assert set(np.flatnonzero(x[:448])) == expected
    assert x.dtype == np.float32 and x.shape == (449,) and x[448] == 0.5
    assert grid_index(7, 7, 6) == 447


def test_reachable_observation_invariants_and_horizon():
    for seed in range(20):
        env = Env(make_tape(seed, GameConfig(bomb_probability=0)))
        rng = random.Random(seed + 1000)
        while not env.done:
            result = env.step(rng.randrange(3))
            x = result.observation
            assert len(env.state.objects) <= 7
            assert np.isin(x[:448], [0, 1]).all()
            assert x[:448].sum() == 1 + len(env.state.objects)
            assert len({o.row for o in env.state.objects}) == len(env.state.objects)
            assert np.count_nonzero(x.reshape(-1)[:392].reshape(7, 8, 7)[:, :, 0]) == 0
        assert env.elapsed_ticks == 256 and env.observe()[448] == 0


def test_different_actions_same_executed_spawn_stages_and_immutable_tape():
    tape = make_tape(17)
    original = tuple(tape)
    envs = [Env(tape), Env(tape)]
    spawns = [{}, {}]
    for i, env in enumerate(envs):
        rng = random.Random(i)
        while not env.done:
            result = env.step(rng.randrange(3))
            # Finished steps skip the entire spawn stage.
            if not env.done:
                spawns[i][env.elapsed_ticks] = tuple(e.object for e in result.events if e.kind == "spawn")
    common = spawns[0].keys() & spawns[1].keys()
    assert common
    assert all(spawns[0][tick] == spawns[1][tick] for tick in common)
    assert tape == original
    assert tape == make_tape(17)  # independent action RNG never perturbs scene generation


def test_byte_identical_replay_and_independent_review_simulator():
    path = Path(__file__).resolve().parents[1] / "ricochet_sim.py"
    spec = importlib.util.spec_from_file_location("review_sim", path)
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)
    for seed in range(20):
        tape = make_tape(seed)
        assert tape == tuple(review.make_tape(seed, review.Cfg()))
        env = Env(tape)
        other = Env(tape)
        old = review.Env(tape, review.Cfg())
        rng = random.Random(seed + 1234)
        while not env.done:
            action = rng.randrange(3)
            result, replay = env.step(action), other.step(action)
            expected = old.step(action)
            assert (result.reward, result.terminated, result.truncated) == expected
            assert result.events == replay.events
            assert result.observation.tobytes() == replay.observation.tobytes()
            assert result.observation.tobytes() == np.asarray(review.observe(old.state()), dtype=np.float32).tobytes()
            assert env.score == old.score
