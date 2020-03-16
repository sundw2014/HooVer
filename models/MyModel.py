import numpy as np
T = 10
state_start = np.array([1,2])
state_range = np.array([1,1])
def is_unsafe(state):
    if np.linalg.norm(state) > 4:
        return 1. # return unsafe if norm of the state is greater than 4.
    return 0. # return safe otherwise.
def step_forward(state):
    # The input state variable contains the state of the system, the current time step, and the isunafe flag, i.e. state = system_state + [t, is_unsafe(system_state)]
    system_state = state[:-2] # extract the state of the system
    t = state[-2] # extract the current time step
    system_state = np.array(system_state)
    system_state += 1.0 * np.random.randn(len(system_state)) # a normally distributed increment
    system_state = system_state.tolist()
    t += 1 # increase the time step by 1
    return system_state + [t, is_unsafe(system_state)] # return the new state

