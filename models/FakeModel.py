# Author: Negin & Dawei

import numpy as np
import sys
sys.path.append('..')
from utils.general_utils import llist_generator
import time

from llist import dllist, dllistnode

T = 1

num_dims = 2
state_start = np.array([0,] * num_dims)
state_range = np.array([1,] * num_dims)

s = None

def is_unsafe(state):
    prob = get_prob(state)
    seed = int(time.time()*100000) % (2**32)
    np.random.seed(seed)
    return np.random.choice([0., 1.], p=[1-prob, prob])

def get_initial_state(seed):
    np.random.seed(np.abs(seed) % (2**32))
    state = np.random.rand(len(state_start)) * state_range + state_start
    state = state.tolist()
    t = 1.
    #print('seed = '+str(seed)+', '+'state = '+str(state))
    return state + [t, is_unsafe(state)]

def get_prob(state):
    center = state_start + 0.5 * state_range
    r = np.sqrt(np.sum((state-center)**2))
    #prob = min(1.0, (np.exp(-1.0 * r ** 2 / s) + 0.1 * np.exp(-1.0 * r **2))/1.1)
    prob = min(1.0, np.exp(-1.0 * r ** 2 / s))
    return prob

def step_forward(_state):
    assert len(_state) == num_dims + 2
    state = _state[:-2]
    t = _state[-2]

    if int(t)==1:
        seed = int(time.time()*100000) % (2**32)
        np.random.seed(seed)

    t = t + 1

    return state + [t, is_unsafe(state)]
