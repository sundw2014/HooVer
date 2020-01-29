import numpy as np
import math
import sys
sys.path.append('..')
from utils.general_utils import map_to_cube, map_to_bounds, map_cell_to_bounds, llist_generator
import time

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

simulator = None
state_start = None
state_range = None
index_non_const_dims = None
T = None

def set_simulator(module):
    global simulator, state_start, state_range, index_non_const_dims, T
    simulator = module
    state_start = simulator.state_start.astype('float')
    state_range = simulator.state_range.astype('float')
    index_non_const_dims = np.where(state_range>0)[0]
    T = simulator.T

def get_mch_as_mf(batch_size):
    MCh = MarkovChain()
    reward_function = lambda x, z: MCh.run_markov_chain(x, math.ceil(z))
    fidel_cost_function = lambda z: 1
    fidel_dim = 1
    fidel_bounds = np.array([(1,T)] * fidel_dim)
    return MFMarkovChain(reward_function, MCh.init_set_domain_bounds, MCh.init_set_domain_dim, fidel_cost_function,
                         fidel_bounds, fidel_dim, batch_size), MCh
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def get_mch():
    return MarkovChain()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def get_full_state(_state):
    state = state_start.copy()
    state[index_non_const_dims] = _state.reshape(-1)
    return state

class MarkovChain(object):
    """
    Markov chain simulation class for n car platoon with policy.
    """
    def __init__(self):

        """
        time_hor = predefined time horizon for markov chain
        unsafe_rule = rule for specifying unsafe set
        """

        self.init_set_domain_dim = len(index_non_const_dims)
        bounds = np.stack([state_start[index_non_const_dims], state_start[index_non_const_dims] + state_range[index_non_const_dims]]).T
        self.init_set_domain_bounds = bounds

        self.cnt_queries = 0
    # -----------------------------------------------------------------------------

    def run_markov_chain(self, init_state, time_hor):

        self.cnt_queries += 1

        state = get_full_state(init_state).tolist()
        state += [1, simulator.is_unsafe(state)]

        unsafe = state[-1] > 0
        reward = 0

        for step in range(time_hor - 1):
            if unsafe:
                break
            else:
                state = simulator.step_forward(state)
                unsafe = state[-1] > 0

        if unsafe:
            reward = 1

        return reward


class MFMarkovChain(object):
    """
        Markov chain simulation class for n-car platoon with policy.
    """

    def __init__(self, reward_function, domain_bounds, domain_dim, fidel_cost_function, fidel_bounds, fidel_dim, batch_size):

        self.reward_function = reward_function
        self.domain_bounds = domain_bounds
        self.domain_dim = domain_dim
        self.fidel_cost_function = fidel_cost_function
        self.fidel_bounds = fidel_bounds
        self.fidel_dim = fidel_dim
        self.opt_fidel_cost = self.cost_single(1)
        self.max_iteration = 200
        self.batch_size = batch_size
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
        #Z, X = self.get_unnormalised_coords(Z, X)
        #return self.eval_at_fidel_single_point(Z, X)
        return self.eval_at_fidel_single_point_normalised_average(Z, X, self.batch_size)

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
        return MFMarkovChain(MCh_object.reward_function, MCh_object.domain_bounds, MCh_object.domain_dim, MCh_object.fidel_cost_function, MCh_object.fidel_bounds, MCh_object.fidel_dim, MCh_object.batch_size)
