"""Numerical checks independent of the policy's matrix implementation."""

import math

import numpy as np
import pytest

from ricochet_catch.policy import (
    PARAMETER_COUNT,
    TENSOR_SHAPES,
    NeuralPolicy,
    PolicyError,
    zero_tensors,
)


def test_architecture_and_zero_policy() -> None:
    assert PARAMETER_COUNT == 15_555
    tensors = zero_tensors()
    assert {name: tensor.shape for name, tensor in tensors.items()} == TENSOR_SHAPES
    policy = NeuralPolicy(tensors)
    observation = np.zeros(449, dtype=np.float32)
    observation[448] = 1
    np.testing.assert_array_equal(policy.logits(observation), np.zeros(3, dtype=np.float32))
    assert policy.act(observation) == 0


def test_forward_pass_against_scalar_hand_calculation() -> None:
    tensors = zero_tensors()
    # h1[0] = tanh(2*0.5 + 4*(-0.25)) = 0;
    # h1[1] = tanh(0.5 + 0.5) = tanh(1).
    tensors["W1"][0, :2] = [2, 4]
    tensors["W1"][1, 0] = 1
    tensors["b1"][1] = 0.5
    # Only h2[0] and h2[1] are nonzero; their values are calculable
    # independently with scalar math, without reusing a policy helper.
    tensors["W2"][0, :2] = [123, 2]
    tensors["b2"][0] = -1
    tensors["b2"][1] = 0.25
    tensors["W3"][:, :2] = [[2, -1], [-3, 4], [1, 2]]
    tensors["b3"][:] = [0.5, -0.25, 1]
    observation = np.zeros(449, dtype=np.float32)
    observation[:2] = [0.5, -0.25]
    a = math.tanh(2 * math.tanh(1) - 1)
    b = math.tanh(0.25)
    expected = [2 * a - b + 0.5, -3 * a + 4 * b - 0.25, a + 2 * b + 1]
    policy = NeuralPolicy(tensors)
    logits = policy.logits(observation)
    assert logits.dtype == np.float32
    np.testing.assert_allclose(logits, expected, rtol=2e-6, atol=2e-7)
    assert policy.act(observation) == 2


def test_ties_use_smallest_action_index() -> None:
    tensors = zero_tensors()
    tensors["b3"][:] = [-5, 2, 2]
    assert NeuralPolicy(tensors).act(np.zeros(449, dtype=np.float32)) == 1


def test_input_and_artifact_are_not_mutated_and_weights_are_copied() -> None:
    tensors = zero_tensors()
    tensors["b3"][2] = 1
    policy = NeuralPolicy(tensors, name="fixture")
    tensors["b3"][2] = -1
    observation = np.arange(449, dtype=np.float32)
    saved = observation.copy()
    assert policy.act(observation) == 2
    np.testing.assert_array_equal(observation, saved)
    assert policy.name == "fixture"
    with pytest.raises(ValueError):
        policy.tensors["b3"][0] = 5
    with pytest.raises(TypeError):
        policy.tensors["extra"] = np.zeros(1, dtype=np.float32)


@pytest.mark.parametrize("bad", [
    np.zeros(448, dtype=np.float32),
    np.zeros((449, 1), dtype=np.float32),
    np.zeros(449, dtype=np.float64),
    [0] * 449,
    np.full(449, np.nan, dtype=np.float32),
    np.full(449, np.inf, dtype=np.float32),
])
def test_malformed_observations_are_errors(bad) -> None:
    with pytest.raises(PolicyError):
        NeuralPolicy(zero_tensors()).act(bad)


@pytest.mark.parametrize("issue", ["missing", "extra", "shape", "dtype", "nonfinite"])
def test_malformed_tensor_artifacts_are_errors(issue: str) -> None:
    tensors = zero_tensors()
    if issue == "missing":
        del tensors["b3"]
    elif issue == "extra":
        tensors["other"] = np.zeros(3, dtype=np.float32)
    elif issue == "shape":
        tensors["W1"] = np.zeros((449, 32), dtype=np.float32)
    elif issue == "dtype":
        tensors["b3"] = tensors["b3"].astype(np.float64)
    else:
        tensors["b3"][0] = np.inf
    with pytest.raises(PolicyError):
        NeuralPolicy(tensors)


def test_nonfinite_logits_are_an_error_instead_of_an_action() -> None:
    tensors = zero_tensors()
    tensors["b2"][:] = 1
    tensors["W3"][:] = np.finfo(np.float32).max
    with pytest.raises(PolicyError, match="logits"):
        NeuralPolicy(tensors).act(np.zeros(449, dtype=np.float32))
