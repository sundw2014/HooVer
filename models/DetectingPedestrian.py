# Author: Negin & Dawei

import numpy as np
import sys
sys.path.append('..')
from utils.general_utils import llist_generator
import time

car_pos_range = [50, 100]
car_v_range = [10, 25]
ped_pos_range = [3, 7]
ped_v_range = [1, 2]
states = [car_pos_range, car_v_range, ped_pos_range, ped_v_range]

state_start = np.array([state[0] for state in states])
state_range = np.array([state[1] - state[0] for state in states])

time_step = 0.1 # 0.1 s
brake_acc = 8 # m/s^2

v_error = 1.0

T = 50

def is_unsafe(state):
    if state[0] >= -5 and state[0] <= 0 and state[2] >= 0 and state[2] <= 5:
        return 1.0
    else:
        return 0.

def step_forward(_state):
    assert len(_state) == len(state_start) + 2
    state = list(_state[:-2])
    t = _state[-2]

    t = t + 1

    # real_t = t * time_step

    state[0] -= (state[1] + np.random.randn() * v_error * 0.1) * time_step
    state[2] -= (state[3] + np.random.randn() * v_error) * time_step

    prob = 0.1 * (1 - np.sqrt(state[0] ** 2 + (state[2]*10) ** 2) / np.sqrt(car_pos_range[1] ** 2 + (ped_pos_range[1]*10) ** 2))

    if np.random.rand() < prob:
        # detected
        brake_distance = state[1] ** 2 / (2 * brake_acc)

        time_to_zero = 100000000 if brake_distance < state[0] else (state[1] - np.sqrt(state[1] ** 2 - 2 * brake_acc * state[0])) / brake_acc

        state[2] -= state[3] * time_to_zero
        state[0] = 0.
        state[1] = 0.
    return state + [t, is_unsafe(state)]
