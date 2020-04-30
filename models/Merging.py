import numpy as np
import sys
sys.path.append('..')
from NiMC import NiMC

__all__ = ['Merging']

prob_close = np.array([0.7, 0.3, 0.]) # prob for brake/cruise/speedup when close
prob_fine = np.array([0.15, 0.7, 0.15])
prob_far = np.array([0.1, 0.2, 0.7])
prob_changing_lane = 0.7
v_brake = 1
v_cruise = 3
v_speedup = 5
v_error = 0.1
velocities = [v_brake, v_cruise, v_speedup]
acc = 0.5

threshold_close = 4
threshold_far = 5
threshold_changing_lane_rear = 3
threshold_changing_lane_ahead = 3

num_cars_lane_1 = 3
initial_separation_between_cars = 15
range_of_initial_set_per_car = 5
lane0_initial_speed_range = [0, 1]
lane0_initial_pos = float((num_cars_lane_1 - 1) * initial_separation_between_cars)

state_start = (np.array(list(range(num_cars_lane_1))) * initial_separation_between_cars)[::-1]

state_start = np.array(state_start.tolist() + [lane0_initial_pos, lane0_initial_speed_range[0], 0.])
state_range = np.array([range_of_initial_set_per_car, ]*num_cars_lane_1 + [0., lane0_initial_speed_range[1] - lane0_initial_speed_range[0], 0.])

Theta = np.stack([state_start, state_start + state_range]).T

unsafe_rule = 2
lane0_end = lane0_initial_pos + 20


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

class Merging(NiMC):
    def __init__(self, k=10):
        super(Merging, self).__init__()
        self.set_Theta(Theta)
        self.set_k(k)

    def is_unsafe(self, state):
        lane1 = state[:num_cars_lane_1]
        if state[-1] == 0: # on lane 0
            if state[-3] > lane0_end:
                return True
        else: # changed to lane 1
            lane1.append(state[-3])
            lane1.sort(reverse=True)

        for car in range(1, len(lane1)):
            if lane1[car-1] - lane1[car] < unsafe_rule:
                return True

        return False

    def transition(self, state):
        assert len(state) == num_cars_lane_1 + 3

        state_new = []

        lane1 = state[:num_cars_lane_1]
        if state[-1] == 1: # on lane 1
            lane1.append(state[-3])
            lane1.sort(reverse=True)

        for car in range(len(lane1)):
            ita = np.random.randn() * v_error
            cur_pos = lane1[car] # current car
            if car==0:
                # first car, always cruise
                p = prob_fine
            else:
                # not the first car
                distance = lane1[car-1] - cur_pos
                p = prob_close if distance < threshold_close else prob_fine if distance < threshold_far else prob_far
            v = np.random.choice(velocities, p = p)
            position = cur_pos + v + ita
            state_new.append(position)

        lane_id = state[-1]
        lane0_v = state[-2]
        lane0_pos = state[-3]

        if lane_id == 0: # on lane 0
            ita = np.random.randn() * v_error
            lane0_pos += lane0_v + ita
            if lane0_v < v_speedup:
                lane0_v += acc
            state_new.append(lane0_pos)

            safe_to_change = check_lane(lane0_pos, lane1) # check if it is safe to change the lane
            p_keep = 1 - prob_changing_lane if safe_to_change else 1
            if np.random.rand() < p_keep:
                lane_id = 0.
            else:
                lane_id = 1.

        state_new.append(lane0_v)
        state_new.append(lane_id)

        return state_new
