# Author: Negin

import numpy as np
import math
from utils.general_utils import map_to_cube, map_to_bounds, map_cell_to_bounds, llist_generator
from numpy.core._multiarray_umath import ndarray
import time


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def get_mch_as_mf():
    MCh = MarkovChain()
    reward_function = lambda x, z: MCh.run_markov_chain(x, math.ceil(z))
    fidel_cost_function = lambda z: 1
    fidel_dim = 1
    fidel_bounds = np.array([(20, 50)] * fidel_dim)
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
    Markov chain simulation class for example 1.
    """
    def __init__(self):

        """
        prob_trans = state transition rule in y direction
        time_hor = predefined time horizon for markov chain
        init_set_domain = specifies the bounds for the set of initial states along x and y directions
        unsafe_rule = rule for specifying unsafe set
        """
        self.prob_trans = 0.5
        self.num_cars = 2
        self.min_dist = 5
        self.state_space_max_pos = 100
        # self.time_hor = 20
        self.init_set_domain_dim = self.num_cars
        self.vel_prob = 0.9
        bounds = np.array([(0, 0)] * self.init_set_domain_dim)
        bounds[0][0] = 0
        bounds[0][1] = 2
        bounds[1][0] = 5
        bounds[1][1] = 8
        # bounds[1][0] = 1
        # bounds[1][1] = self.min_dist
        # for j in range(self.num_cars-2):
        #     bounds[j+2][0] = bounds[j+1][1] + 1
        #     bounds[j+2][1] = bounds[j+1][1] + self.min_dist
        self.init_set_domain_bounds = bounds
        self.unsafe_rule = 2*self.min_dist
        self.speeds = np.array([4, 3])

    # -----------------------------------------------------------------------------

    def run_markov_chain(self, init_state, time_hor):
        state_current_ = [0]*self.num_cars
        for j in range(self.num_cars):
            state_current_[j] = init_state[0][j]
        state_current = state_current_
        unsafe = self.is_unsafe(state_current)
        reward = 0

        for step in range(time_hor - 1):

            state_current_ = sorted(state_current_)

            if unsafe:
                break
            elif not unsafe:
                for k in range(self.num_cars):
                    if k == self.num_cars-1:
                        state_current[k] = state_current_[k] + self.speeds[0]
                    else:
                        if (state_current_[k+1]-state_current_[k]) > self.min_dist:
                            rnd = np.random.random()
                            if rnd <= self.vel_prob:
                                state_current[k] = state_current_[k] + self.speeds[0]
                            else:
                                state_current[k] = state_current_[k] + self.speeds[1]
                        else:
                            rnd = np.random.random()
                            if rnd <= self.vel_prob:
                                state_current[k] = state_current_[k] + self.speeds[1]
                            else:
                                state_current[k] = state_current_[k] + self.speeds[0]
                state_current_ = state_current
                state_current_ = sorted(state_current_)
                unsafe = self.is_unsafe(state_current_)

        # for step in range(time_hor - 1):
        #
        #     state_current_ = sorted(state_current_)
        #
        #     if unsafe:
        #         break
        #     elif not unsafe:
        #         for k in range(self.num_cars):
        #             rnd = np.random.random()
        #             if rnd <= self.prob_trans:
        #                 state_current[k] = state_current_[k] + self.speeds[1]
        #             else:
        #                 state_current[k] = state_current_[k] + self.speeds[0]
        #         state_current_ = state_current
        #         state_current_ = sorted(state_current_)
        #         unsafe = self.is_unsafe(state_current_)

        if unsafe:
            reward = 1

        # if unsafe:
        #     reward = 1 + np.random.normal(0, np.sqrt(5), 1)
        # else:
        #     reward = 0 + np.random.normal(0, np.sqrt(5), 1)

        return reward

    # -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

    def is_unsafe(self, state):
        unsafe = False

        for k in range(self.num_cars-1):
            if abs(state[k] - state[k+1]) > self.unsafe_rule:
                unsafe = True
                break

        return unsafe

    # -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class MFMarkovChain(object):
    """
        Markov chain simulation class for example 1.
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

    def get_unnormalised_cell(self, X):
        """ Maps points in the cube to the original space. """
        ret_Cell = None if X is None else map_cell_to_bounds(X, self.domain_bounds)
        return ret_Cell

    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------

    def mf_markov_chain(MCh_object):
        return MFMarkovChain(MCh_object.reward_function, MCh_object.domain_bounds, MCh_object.domain_dim, MCh_object.fidel_cost_function, MCh_object.fidel_bounds, MCh_object.fidel_dim)
