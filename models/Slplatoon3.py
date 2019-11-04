# Author: Negin & Dawei

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
    fidel_bounds = np.array([(3,20)] * fidel_dim)
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
        self.num_cars = 2
        self.init_set_domain_dim = self.num_cars
        self.max_pos = 300
        initial_separation_between_cars = 10
        range_of_initial_set_per_car = 5
        bounds = np.array(list(range(self.num_cars))) * initial_separation_between_cars
        bounds = np.stack([bounds, bounds + range_of_initial_set_per_car]).T
        self.init_set_domain_bounds = bounds[::-1,:]

        self.prob_close = np.array([0.7,0.15,0.15]) # prob for brake/cruise/speedup when close
        self.prob_fine = np.array([0.15,0.7,0.15])
        self.prob_far = np.array([0.15,0.15,0.7])

        self.v_brake = 1
        self.v_cruise = 4
        self.v_speedup = 7
        # self.v_error = 0
        self.v = [self.v_brake, self.v_cruise, self.v_speedup]

        self.threshold_close = 3
        self.threshold_far = 5

        self.unsafe_rule_close = 1
        self.unsafe_rule_far = 20
        # self.offset = 1e-3 # make it continuous at integer points
        self.offset = 0

    # -----------------------------------------------------------------------------

    def run_markov_chain(self, init_state, time_hor, seed=None, offset=False, dump_trace=False, v_error=0):
        # init_state = init_state[0] + self.offset
        init_state = init_state[0] + self.offset * np.array([(-1)**i for i in range(len(init_state[0]))])
        if seed is not None:
            np.random.seed(seed)

        state_current = dllist(init_state)
        unsafe = self.is_unsafe(state_current)
        run_out = self.is_run_out(state_current)
        reward = 0

        if dump_trace: trace = [init_state,]
        for step in range(time_hor - 1):
            # import pdb; pdb.set_trace()
            # print(state_current)
            # step_desc = ''
            if unsafe:
                break
            elif run_out:
                break
            elif (not unsafe) and (not run_out):
                state_old = state_current
                state_current = dllist()
                # update the state
                for car in llist_generator(state_old):
                    ita = np.random.randn() * v_error
                    # ita = 0
                    if car is state_old.first:
                        # first car, always cruise
                        position = car.value + self.v_cruise + ita
                        state_current.append(position)
                        # step_desc += 'first'
                    else:
                        # not the first car
                        distance = car.prev.value - car.value
                        p = self.prob_close if distance < self.threshold_close else self.prob_fine if distance < self.threshold_far else self.prob_far
                        # choice = 0 if distance < self.threshold_close else 1 if distance < self.threshold_far else 2

                        v = np.random.choice(self.v, p = p)
                        position = car.value + v + ita
                        state_current.append(position)
                        # step_desc += ' [%d]'%choice
                # print(step_desc)
                unsafe = self.is_unsafe(state_current)
                run_out = self.is_run_out(state_current)
                if dump_trace: trace.append(np.array(list(state_current)))

        if dump_trace:
            for _ in range(len(trace), time_hor):
                trace.append(np.array([-1,-1]))

        if unsafe:
            reward = 1
        if offset:
            R = np.array([round(x) - x for x in init_state])
            R = np.sqrt(R.dot(R).sum())
            R0 = np.sqrt((0.5 ** 2) * len(init_state))
            reward_offset = (((R - R0) / R0) ** 2 - 1) * 0.1
            reward += reward_offset
        if dump_trace:
            return reward, trace
        else:
            return reward

    def is_unsafe(self, state):
        unsafe = False
        for car in llist_generator(state):
            if car is state.first: continue
            distance = car.prev.value - car.value
            if distance < self.unsafe_rule_close:
                unsafe = True
                break
        # if state[0] - state[-1] > self.unsafe_rule_far:
        #     unsafe = True
        return unsafe

    def is_run_out(self, state):
        run_out = False
        for car in llist_generator(state):
            if self.max_pos - car.value < self.v_speedup:
                run_out = True
                break
        return run_out


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
        # return self.eval_at_fidel_single_point_normalised_average(Z, X, 1)

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
