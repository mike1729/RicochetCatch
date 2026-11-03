"""The fixed, stateless float32 policy used by the experiment.

This module deliberately contains no game rules, action correction, or training.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np


TENSOR_SHAPES: dict[str, tuple[int, ...]] = {
    "W1": (32, 449),
    "b1": (32,),
    "W2": (32, 32),
    "b2": (32,),
    "W3": (3, 32),
    "b3": (3,),
}
OBSERVATION_SIZE = 449
PARAMETER_COUNT = sum(int(np.prod(shape)) for shape in TENSOR_SHAPES.values())


class PolicyError(ValueError):
    """A policy artifact or observation cannot be used for inference."""


def zero_tensors() -> dict[str, np.ndarray]:
    """Return a fresh, writable, float32 zero initialization."""
    return {name: np.zeros(shape, dtype=np.float32) for name, shape in TENSOR_SHAPES.items()}


def validate_tensors(tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Validate and defensively copy all six tensors into immutable arrays.

    Requiring float32, rather than accepting an implicit conversion, makes the
    inference dtype an explicit artifact property. The assignment parser performs
    its checked conversion before calling this function.
    """
    if not isinstance(tensors, Mapping) or set(tensors) != set(TENSOR_SHAPES):
        raise PolicyError("tensors must contain exactly W1, b1, W2, b2, W3, b3")
    result: dict[str, np.ndarray] = {}
    for name, shape in TENSOR_SHAPES.items():
        value = tensors[name]
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float32):
            raise PolicyError(f"{name} must be a float32 NumPy array")
        if value.shape != shape:
            raise PolicyError(f"{name} must have shape {shape}")
        if not np.isfinite(value).all():
            raise PolicyError(f"{name} contains a nonfinite value")
        copied = np.array(value, dtype=np.float32, order="C", copy=True)
        copied.flags.writeable = False
        result[name] = copied
    return result


class NeuralPolicy:
    """449 → 32 → 32 → 3 tanh network with deterministic argmax actions."""

    def __init__(self, tensors: Mapping[str, np.ndarray], name: str = "network") -> None:
        self.tensors = MappingProxyType(validate_tensors(tensors))
        self.name = name

    def logits(self, observation: np.ndarray) -> np.ndarray:
        """Compute the three logits using only NumPy float32 arithmetic."""
        if not isinstance(observation, np.ndarray) or observation.dtype != np.dtype(np.float32):
            raise PolicyError("observation must be a float32 NumPy array")
        if observation.shape != (OBSERVATION_SIZE,):
            raise PolicyError(f"observation must have shape ({OBSERVATION_SIZE},)")
        if not np.isfinite(observation).all():
            raise PolicyError("observation contains a nonfinite value")
        tensors = self.tensors
        # Artifact and observation validation precede every forward pass. Any
        # arithmetic overflow must still become an explicit implementation error.
        with np.errstate(over="ignore", invalid="ignore"):
            h1 = np.tanh(tensors["W1"] @ observation + tensors["b1"])
            h2 = np.tanh(tensors["W2"] @ h1 + tensors["b2"])
            logits = tensors["W3"] @ h2 + tensors["b3"]
        if logits.dtype != np.dtype(np.float32) or not np.isfinite(logits).all():
            raise PolicyError("forward pass produced nonfinite or non-float32 logits")
        return logits

    def act(self, observation: np.ndarray) -> int:
        """Choose stay/left/right; NumPy argmax breaks ties by first index."""
        return int(np.argmax(self.logits(observation)))

    __call__ = act
