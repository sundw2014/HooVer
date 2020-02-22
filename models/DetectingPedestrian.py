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

time_step = 0.25 # 0.1 s
brake_acc = 8 # m/s^2

v_error = 0.

# T = 20
T = 3

def random_initialization(seed, initial_states=None):
    if initial_states is not None:
        state = initial_states
    else:
        np.random.seed(np.abs(seed) % (2**32))
        state = np.random.rand(len(state_start)) * state_range + state_start
        state = state.tolist()
    t = 1.
    print('seed = '+str(seed)+', '+'state = '+str(state))
    return state + [t, is_unsafe(state)]

def is_unsafe(state):
    if state[0] >= -5 and state[0] <= 0 and state[2] >= -1 and state[2] <= 1:
        return 1.0
    else:
        return 0.

def step_forward(_state):
    assert len(_state) == len(state_start) + 2
    state = list(_state[:-2])
    t = _state[-2]

    if int(t)==1:
        seed = int(time.time()*100000) % (2**32)
        np.random.seed(seed)

    t = t + 1

    # real_t = t * time_step

    # this is an unsafe lidar model
    def lidar_prob(theta, r):
        s = 1e-7
        theta_broken = 0.08
        r_max = 10000
        prob = (1 - np.exp(-1.0 * (theta - theta_broken) ** 2 / s))# * ((r - r_max) ** 2 / (r_max ** 2))
        return prob

    if state[0] > 0:
        theta = np.arctan(state[2] / state[0])
        r = np.sqrt(state[2] ** 2 + state[0] ** 2)
        prob = min(max(lidar_prob(theta, r), 0), 1)
    else:
        prob = 0

    if np.random.rand() < prob:
        # detected
        brake_distance = state[1] ** 2 / (2 * brake_acc)

        time_to_zero = 100000000 if brake_distance < state[0] else (state[1] - np.sqrt(state[1] ** 2 - 2 * brake_acc * state[0])) / brake_acc

        state[2] -= state[3] * time_to_zero
        state[0] = 0.
        state[1] = 0.

    state[0] -= (state[1] + np.random.randn() * v_error * 0.1) * time_step
    state[2] -= (state[3] + np.random.randn() * v_error) * time_step

    return state + [t, is_unsafe(state)]
