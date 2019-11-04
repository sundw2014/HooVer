# Author: Negin

import numpy as np
import math
from utils.general_utils import map_to_cube, map_to_bounds, map_cell_to_bounds, llist_generator
from numpy.core._multiarray_umath import ndarray
import time

from llist import dllist, dllistnode

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def get_mch_as_mf():
    MCh = MarkovChain()
    reward_function = lambda x, z: MCh.run_markov_chain(x, math.ceil(z))
    fidel_cost_function = lambda z: 1
    fidel_dim = 1
    fidel_bounds = np.array([(3, 20)] * fidel_dim)
    return MFMarkovChain(reward_function, MCh.init_set_domain_bounds, MCh.init_set_domain_dim, fidel_cost_function,
                         fidel_bounds, fidel_dim)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def get_mch():
    return MarkovChain()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class MarkovChain(object):
    """
    Markov chain simulation class for n car platoon with policy.
    """
    def __init__(self):

        """
        time_hor = predefined time horizon for markov chain
        unsafe_rule = rule for specifying unsafe set
        """
        self.num_cars = 2 # TOTAL
        self.num_lanes = 2
        self.init_set_domain_dim = self.num_cars
        initial_avg_distance = 10
        range_of_initial_position = 5
        bounds = np.array(list(range(self.num_cars))) * initial_avg_distance
        bounds = np.stack([bounds, bounds + range_of_initial_position]).T
        self.init_set_domain_bounds = bounds[::-1,:]

        self.prob_close = np.array([0.7, 0.15, 0.15]) # prob for brake/cruise/speedup when close
        self.prob_fine = np.array([0.15, 0.7, 0.15])
        self.prob_far = np.array([0.15, 0.15, 0.7])
        self.prob_changing_lane = 0.15

        self.v_brake = 1
        self.v_cruise = 3
        self.v_speedup = 5
        self.v = [self.v_brake, self.v_cruise, self.v_speedup]

        self.threshold_close = 4
        self.threshold_far = 5
        self.threshold_changing_lane_rear = 3
        self.threshold_changing_lane_ahead = 3

        self.unsafe_rule = 3

        self.offset = 1e-3
        ### FIXME: max_pos
    # -----------------------------------------------------------------------------

    def check_lane(self, car, lane):
        if len(lane) == 0:
            return True
        distance_ahead = lane-car
        distance_ahead[distance_ahead<0] = np.Inf
        distance_ahead = distance_ahead.min()

        distance_rear = car-lane
        distance_rear[distance_rear<=0] = np.Inf
        distance_rear = distance_rear.min()

        return (distance_rear > self.threshold_changing_lane_rear) and (distance_ahead > self.threshold_changing_lane_ahead)

    def run_markov_chain(self, init_state, time_hor, seed=None, offset=False, dump_trace=False, v_error=0):
        assert dump_trace == False
        # init_state = init_state[0]
        init_state = init_state[0] + self.offset * np.array([(-1)**i for i in range(len(init_state[0]))])

        if seed is not None:
            np.random.seed(seed)
        state_current = []
        # from IPython import embed; embed()
        state_current.append(init_state)
        for i in range(self.num_lanes-1):
            state_current.append([])
        unsafe = self.is_unsafe(state_current)
        reward = 0

        for step in range(time_hor - 1):
            if unsafe:
                break
            else:
                state_old = state_current
                state_current = [[] for _ in range(self.num_lanes)]
                # update the state
                for lane in range(len(state_old)):
                    for car in range(len(state_old[lane])):
                        ita = np.random.randn() * v_error
                        cur_pos = state_old[lane][car] # current car
                        safe_left = False
                        safe_right = False
                        if lane is not 0:
                            safe_left = self.check_lane(cur_pos, state_old[lane-1]) # check if it is safe to change the lane
                        if lane is not (len(state_old)-1):
                            safe_right = self.check_lane(cur_pos, state_old[lane+1])
                        p_left = self.prob_changing_lane if safe_left else 0
                        p_right = self.prob_changing_lane if safe_right else 0
                        p_keep = 1.0 - p_left - p_right
                        choice = np.random.choice(a = [0,1,2], p=[p_left, p_right, p_keep])
                        if choice == 0:
                            state_current[lane-1].append(cur_pos + self.v_cruise + ita)
                        elif choice == 1:
                            state_current[lane+1].append(cur_pos + self.v_cruise + ita)
                        elif choice==2:
                            if car==0:
                                # first car, always cruise
                                p = self.prob_fine
                            else:
                                # not the first car
                                distance = state_old[lane][car-1] - cur_pos
                                p = self.prob_close if distance < self.threshold_close else self.prob_fine if distance < self.threshold_far else self.prob_far
                            v = np.random.choice(self.v, p = p)
                            position = cur_pos + v + ita
                            state_current[lane].append(position)
                        else:
                            raise ValueError()

                state_current = [np.array(sorted(lane, reverse=True)) for lane in state_current]
                unsafe = self.is_unsafe(state_current)

        if unsafe:
            reward = 1
        if offset:
            R = np.array([round(x) - x for x in init_state])
            R = np.sqrt(R.dot(R).sum())
            R0 = np.sqrt((0.5 ** 2) * len(init_state))
            reward_offset = ((R - R0) / R0) ** 2 - 1
            reward += reward_offset
        return reward

    def is_unsafe(self, state):
        unsafe = False
        for lane in state:
            for car in range(1, len(lane)):
                if lane[car-1] - lane[car] < self.unsafe_rule:
                    unsafe = True
                    return unsafe

        return unsafe


class MFMarkovChain(object):
    """
        Markov chain simulation class for n-car platoon with policy.
    """

    def __init__(self, reward_function, domain_bounds, domain_dim, fidel_cost_function, fidel_bounds, fidel_dim):

        self.reward_function = reward_function
        self.domain_bounds = domain_bounds
        self.domain_dim = domain_dim
        self.fidel_cost_function = fidel_cost_function
        self.fidel_bounds = fidel_bounds
        self.fidel_dim = fidel_dim
        self.opt_fidel_cost = self.cost_single(1)
        self.max_iteration = 200
        # self.opt_fidel_cost = self.cost_single_average(1)

    # -----------------------------------------------------------------------------

    def cost_single_average(self, Z):
        """ Evaluates cost at a single point. """
        t1 = time.time()
        d = self.domain_dim
        X = np.array([0.5] * d)
        self.eval_at_fidel_single_point_normalised_average(Z, X, self.max_iteration)
        t2 = time.time()
        return t2-t1

    # -----------------------------------------------------------------------------

    def cost_single(self, Z):
        """ Evaluates cost at a single point. """
        return self.eval_fidel_cost_single_point_normalised(Z)

    # -----------------------------------------------------------------------------

    def eval_at_fidel_single_point(self, Z, X):
        """ Evaluates X at the given Z at a single point. """
        Z = np.array(Z).reshape((1, self.fidel_dim))
        X = np.array(X).reshape((1, self.domain_dim))
        return float(self.reward_function(X, Z))

    # -----------------------------------------------------------------------------

    def eval_fidel_cost_single_point(self, Z):
        """ Evaluates the cost function at a single point. """
        return float(self.fidel_cost_function(Z))

    # -----------------------------------------------------------------------------

    def eval_at_fidel_single_point_normalised(self, Z, X):
        """ Evaluates X at the given Z at a single point using normalised coordinates. """
        Z, X = self.get_unnormalised_coords(Z, X)
        return self.eval_at_fidel_single_point(Z, X)

    # -----------------------------------------------------------------------------

    def eval_at_fidel_single_point_normalised_average(self, Z, X, max_iteration):
        """ Evaluates X at the given Z at a single point using normalised coordinates. """
        Z, X = self.get_unnormalised_coords(Z, X)

        mean_value = 0

        for i in range(max_iteration):
            value = self.eval_at_fidel_single_point(Z, X)
            mean_value = mean_value + value

        mean_value = mean_value/max_iteration

        return mean_value

    # -----------------------------------------------------------------------------

    def eval_fidel_cost_single_point_normalised(self, Z):
        """ Evaluates the cost function at a single point using normalised coordinates. """
        Z, _ = self.get_unnormalised_coords(Z, None)
        return self.eval_fidel_cost_single_point(Z)

    # -----------------------------------------------------------------------------

    def get_normalised_coords(self, Z, X):
        """ Maps points in the original space to the cube. """
        ret_Z = None if Z is None else map_to_cube(Z, self.fidel_bounds)
        ret_X = None if X is None else map_to_cube(X, self.domain_bounds)
        return ret_Z, ret_X

    # -----------------------------------------------------------------------------

    def get_unnormalised_coords(self, Z, X):
        """ Maps points in the cube to the original space. """
        ret_X = None if X is None else map_to_bounds(X, self.domain_bounds)
        ret_Z = None if Z is None else map_to_bounds(Z, self.fidel_bounds)
        return ret_Z, ret_X

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    def get_unnormalised_cell(self, X):
        """ Maps points in the cube to the original space. """
        ret_Cell = None if X is None else map_cell_to_bounds(X, self.domain_bounds)
        return ret_Cell

    # -----------------------------------------------------------------------------

    def mf_markov_chain(MCh_object):
        return MFMarkovChain(MCh_object.reward_function, MCh_object.domain_bounds, MCh_object.domain_dim, MCh_object.fidel_cost_function, MCh_object.fidel_bounds, MCh_object.fidel_dim)
