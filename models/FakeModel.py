# Author: Negin & Dawei

import numpy as np
import sys
sys.path.append('..')
from utils.general_utils import llist_generator
import time

from llist import dllist, dllistnode

num_dims = 2
state_start = np.array([0,] * num_dims)
state_range = np.array([1,] * num_dims)

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

#ss = np.logspace(np.log(0.0004)/np.log(2), np.log(0.001)/np.log(2), num=10, base=2)
#ss = np.logspace(np.log(1e-6)/np.log(2), np.log(1e-3)/np.log(2), num=10, base=2)
#ss = np.logspace(np.log(ss[6])/np.log(2), np.log(ss[7])/np.log(2), num=10, base=2)
#ss = np.logspace(np.log(ss[5])/np.log(2), np.log(1e-3)/np.log(2), num=10, base=2)
#ss = np.logspace(np.log(1e-4)/np.log(2), np.log(4e-4)/np.log(2), num=10, base=2)
ss = np.logspace(np.log(0.00015)/np.log(2), np.log(4e-4)/np.log(2), num=10, base=2)

def is_unsafe(state):
    if sys.argv[1] == '--port':
        i = int(sys.argv[2]) - 9000
    else:
        i = int(sys.argv[1]) - 1
    prob = get_prob(state, i)
    seed = int(time.time()*100000) % (2**32)
    np.random.seed(seed)
    return np.random.choice([0., 1.], p=[1-prob, prob])

def get_initial_state(seed):
    np.random.seed(np.abs(seed) % (2**32))
    state = np.random.rand(len(state_start)) * state_range + state_start
    state = state.tolist()
    #t = 1.
    #print('seed = '+str(seed)+', '+'state = '+str(state))
    return state# + [t, is_unsafe(state)]


def get_prob(state, i):
    center = state_start + 0.5 * state_range
    s = ss[i]
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
