"""The unprocessed, channel-last 449-element float32 policy observation."""
import numpy as np
from .env import GEM, HORIZON, PolicyState, validate_state

OBSERVATION_SIZE = 449


def grid_index(row, column, channel):
    for value, limit in ((row, 8), (column, 8), (channel, 7)):
        if type(value) is not int or not 0 <= value < limit:
            raise ValueError("grid coordinates out of bounds")
    return ((row * 8 + column) * 7 + channel)


def observe(state: PolicyState):
    validate_state(state)
    x = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    x[grid_index(7, state.paddle_column, 0)] = 1
    for obj in state.objects:
        channel = (1 if obj.kind == GEM else 4) + obj.vx + 1
        x[grid_index(obj.row, obj.column, channel)] = 1
    x[448] = np.float32(state.remaining_ticks / HORIZON)
    return x
