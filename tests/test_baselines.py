import importlib.util
import itertools
from pathlib import Path

import pytest

from ricochet_catch import baselines as b
from ricochet_catch.env import Env, Object, PolicyState, make_tape
from ricochet_catch.evaluate import run_episode


def state(paddle=3, objects=(), remaining=248):
    return PolicyState(paddle, tuple(Object(*obj) for obj in objects), remaining)


def test_reactive_and_prediction_fixture():
    s = state(objects=(("gem", 4, 6, 1),))
    for policy, expected_score in ((b.chase, 0), (b.predict_greedy_safe, 1), (b.reference, 1)):
        env = Env((None,) * 256, initial_state=s, spawning=False)
        actions = []
        for _ in range(3):
            action = policy(env.state)
            actions.append(action)
            env.step(action)
        assert env.score == expected_score
        if policy is b.chase:
            assert actions == [2, 2, 2]


def test_lowest_gem_selected_and_action_ties_follow_indices():
    s = state(objects=(("gem", 1, 0, -1), ("gem", 6, 4, 0)))
    assert b.chase(s) == 2
    assert b.chase_safe(s) == 2
    assert b.chase_safe(state(objects=(("bomb", 6, 3, 0),))) == 1
    assert b.reference(state()) == 0


def test_safety_reflects_bombs_and_lead_does_not_predict_landings(monkeypatch):
    s = state(6, (("bomb", 6, 7, 1), ("gem", 5, 7, 1)))
    assert b.safe_actions(s) == (1, 2)
    assert b.one_step_lead_safe(s) == 2
    monkeypatch.setattr(b, "landing", lambda *args: (_ for _ in ()).throw(AssertionError("landing lookup called")))
    assert b.one_step_lead_safe(s) == 2


def test_conflict_sacrifices_gem():
    s = state(objects=(("bomb", 6, 4, 0), ("gem", 5, 5, 0)))
    for policy in (b.chase_safe, b.one_step_lead_safe, b.predict_greedy_safe, b.reference):
        env = Env((None,) * 256, initial_state=s, spawning=False)
        for _ in range(2):
            env.step(policy(env.state))
        assert not env.terminated and env.score == 0


def test_all_immediate_bomb_cases_have_safe_action():
    for paddle, column, vx in itertools.product(range(8), range(8), (-1, 0, 1)):
        s = state(paddle, (("bomb", 6, column, vx),))
        safe = b.safe_actions(s)
        assert safe
        for action in safe:
            env = Env((None,) * 256, initial_state=s, spawning=False)
            assert not env.step(action).terminated


def test_reference_matches_exhaustive_visible_object_planning():
    s = state(objects=(("bomb", 6, 4, 0), ("gem", 5, 2, 0), ("gem", 4, 1, 0)), remaining=3)
    candidates = []
    for actions in itertools.product(range(3), repeat=3):
        env = Env((None,) * 256, initial_state=s, spawning=False)
        for action in actions:
            if env.done:
                break
            env.step(action)
        if not env.terminated:
            candidates.append((env.score, -actions[0]))
    assert b.reference(s) == -max(candidates)[1]


def test_random_rng_is_separate_and_reproducible():
    tape = make_tape(17)
    a, c = b.RandomPolicy(123), b.RandomPolicy(123)
    assert [a(state()) for _ in range(50)] == [c(state()) for _ in range(50)]
    assert tape == make_tape(17)
    assert not hasattr(state().objects, "tape")
    assert Object._fields == ("kind", "row", "column", "vx")
    with pytest.raises(TypeError):
        Object("gem", 2, 3, 1, True)


def test_independent_review_controller_parity():
    path = Path(__file__).parents[1] / "ricochet_sim.py"
    spec = importlib.util.spec_from_file_location("review_sim", path)
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)
    cfg = review.Cfg()
    for name in ("stay", "random", "chase", "chase_safe", "predict_greedy_safe", "reference"):
        for index in range(12):
            expected = review.run_episode(review.POLICIES[name](index), review.make_tape(index, cfg), cfg)
            actual = run_episode(b.make_baseline(name, index), make_tape(index), record=False)
            assert (actual.score, actual.length, actual.terminated) == (expected["score"], expected["length"], expected["bomb"])
