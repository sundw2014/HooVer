import numpy as np
import sys
sys.path.append('..')
from NiMC import NiMC

__all__ = ['DetectingPedestrian']

# some constants
car_pos_range = [50, 100]
car_v_range = [10, 25]
ped_pos_range = [3, 7]
ped_v_range = [1, 2]
Theta = [car_pos_range, car_v_range, ped_pos_range, ped_v_range]

time_step = 0.25 # 0.25 s
brake_acc = 8 # m/s^2

v_error = 0.

s = 5e-6
theta_broken = 0.08
r_max = 500


class DetectingPedestrian(NiMC):
    def __init__(self, k=10):
        super(DetectingPedestrian, self).__init__()
        self.set_Theta(Theta)
        self.set_k(k)

    def is_unsafe(self, state):
        if state[0] >= -5 and state[0] <= 0 and state[2] >= -1 and state[2] <= 1:
            return True
        else:
            return False

    def transition(self, state):
        # print(_state)
        assert len(state) == len(Theta)

        # this is an unsafe lidar model
        def lidar_prob(theta, r):
            prob = (1 - np.exp(-1.0 * (theta - theta_broken) ** 2 / s)) * ((r - r_max) ** 2 / (r_max ** 2))
            return prob

        if state[0] > 0:
            theta = np.arctan(state[2] / state[0])
            r = np.sqrt(state[2] ** 2 + state[0] ** 2)
            prob = min(max(lidar_prob(theta, r), 0), 1)
        else:
            prob = 0

        if np.random.rand() < prob:
            # print('detected')
            # detected
            brake_distance = state[1] ** 2 / (2 * brake_acc)

            time_to_zero = 100000000 if brake_distance < state[0] else (state[1] - np.sqrt(state[1] ** 2 - 2 * brake_acc * state[0])) / brake_acc

            state[2] -= state[3] * time_to_zero
            state[0] = 0.
            state[1] = 0.

        state[0] -= (state[1] + np.random.randn() * v_error * 0.1) * time_step
        state[2] -= (state[3] + np.random.randn() * v_error) * time_step

        return state
