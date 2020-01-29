# Author: Negin & Dawei

import numpy as np
import sys
sys.path.append('..')
from utils.general_utils import map_to_cube, map_to_bounds, map_cell_to_bounds, llist_generator
import time

prob_close = np.array([0.9, 0.05, 0.05]) # prob for brake/cruise/speedup when close
prob_fine = np.array([0.05, 0.9, 0.05])
prob_far = np.array([0.05, 0.05, 0.9])
prob_changing_lane = 0.15
v_brake = 1
v_cruise = 3
v_speedup = 5
v_error = 0.1
velocities = [v_brake, v_cruise, v_speedup]

threshold_close = 4
threshold_far = 5
threshold_changing_lane_rear = 3
threshold_changing_lane_ahead = 3

num_cars = 3
num_lanes = 3
initial_separation_between_cars = 10
range_of_initial_set_per_car = 5
state_start = np.concatenate([(np.array(list(range(num_cars)) * num_lanes) * initial_separation_between_cars)[::-1], np.array(list(range(num_lanes)) * num_cars).reshape(num_cars, num_lanes).T.reshape(-1).astype('float')])
state_range = np.array([range_of_initial_set_per_car, ] * num_cars * num_lanes + [0, ] * num_cars * num_lanes)

unsafe_rule = 2

T = 10

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

def check_lane(car, lane):
    if len(lane) == 0:
        return True
    lane = np.array(lane)
    distance_ahead = lane-car
    distance_ahead[distance_ahead<0] = np.Inf
    distance_ahead = distance_ahead.min()

    distance_rear = car-lane
    distance_rear[distance_rear<=0] = np.Inf
    distance_rear = distance_rear.min()

    return (distance_rear > threshold_changing_lane_rear) and (distance_ahead > threshold_changing_lane_ahead)

def is_unsafe(state):
    state = map2lf(state)
    for lane in state:
        for car in range(1, len(lane)):
            if lane[car-1] - lane[car] < unsafe_rule:
                return 1.
    return 0.

def map2lf(state): # map -> linear format
    assert len(state) == num_lanes*num_cars*2
    state_x = state[:len(state)//2]
    state_y = state[len(state)//2::]
    state = [[] for _ in range(num_lanes)]
    for x,y in zip(state_x, state_y):
        state[int(y)].append(x)
    for lane in state:
        lane.sort(reverse = True)
    return state

def lf2map(state): # linear format -> map
    state_x = []
    state_y = []
    for lane in range(len(state)):
        for car in range(len(state[lane])):
            state_x.append(state[lane][car])
            state_y.append(float(lane))
    return state_x + state_y

def step_forward(_state):
    # import pdb; pdb.set_trace()
    assert len(_state) == num_cars*num_lanes*2 + 2
    state = _state[:-2]
    t = _state[-2]
    if int(t)==1:
        seed = int(time.time()*100000) % (2**32)
        np.random.seed(seed)

    t = t + 1

    state = map2lf(state)

    state_old = state
    state_new = [[] for _ in range(num_lanes)]
    # update the state
    for lane in range(len(state_old)):
        for car in range(len(state_old[lane])):
            ita = np.random.randn() * v_error
            cur_pos = state_old[lane][car] # current car
            safe_left = False
            safe_right = False
            if lane is not 0:
                safe_left = check_lane(cur_pos, state_old[lane-1]) # check if it is safe to change the lane
            if lane is not (len(state_old)-1):
                safe_right = check_lane(cur_pos, state_old[lane+1])
            p_left = prob_changing_lane if safe_left else 0
            p_right = prob_changing_lane if safe_right else 0
            p_keep = 1.0 - p_left - p_right
            choice = np.random.choice(a = [0,1,2], p=[p_left, p_right, p_keep])
            if choice == 0:
                state_new[lane-1].append(cur_pos + v_cruise + ita)
            elif choice == 1:
                state_new[lane+1].append(cur_pos + v_cruise + ita)
            elif choice==2:
                if car==0:
                    # first car, always cruise
                    p = prob_fine
                else:
                    # not the first car
                    distance = state_old[lane][car-1] - cur_pos
                    p = prob_close if distance < threshold_close else prob_fine if distance < threshold_far else prob_far
                v = np.random.choice(velocities, p = p)
                position = cur_pos + v + ita
                state_new[lane].append(position)
    for lane in state_new:
        lane.sort(reverse = True)
    state = lf2map(state_new)
    us_flag = is_unsafe(state)
    return state + [t, us_flag]
