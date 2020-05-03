import numpy as np
import sys
sys.path.append('..')
from NiMC import NiMC

__all__ = ['Mlplatoon18']

# some constants
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

num_cars = 9
num_lanes = 2
initial_separation_between_cars = 10
range_of_initial_set_per_car = 5
state_start = np.concatenate([(np.array(list(range(num_cars)) * num_lanes) * initial_separation_between_cars)[::-1], np.array(list(range(num_lanes)) * num_cars).reshape(num_cars, num_lanes).T.reshape(-1).astype('float')])
state_range = np.array([range_of_initial_set_per_car, ] * num_cars * num_lanes + [0, ] * num_cars * num_lanes)

# fix the initial position for some of the cars
state_range[np.array([0,2,4,6,8,10,12,14,16,17])] = 0

Theta = np.stack([state_start, state_start + state_range]).T
unsafe_rule = 2

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

class Mlplatoon18(NiMC):
    def __init__(self, k=9):
        super(Mlplatoon18, self).__init__()
        self.set_Theta(Theta)
        self.set_k(k)

    def is_unsafe(self, state):
        state = map2lf(state)
        for lane in state:
            for car in range(1, len(lane)):
                if lane[car-1] - lane[car] < unsafe_rule:
                    return True
        return False

    def transition(self, state):
        assert len(state) == num_cars*num_lanes*2

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
        return state
