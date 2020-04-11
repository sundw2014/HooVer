import numpy as np
import math
import sys
sys.path.append('..')
from utils.general_utils import map_to_cube, map_to_bounds, map_cell_to_bounds, llist_generator
import time

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

class MFNiMC(object):
    """
        This class is a wrapper for NiMC in order to use MFHOO.
        This class normalizes the domain such that length of each dim after normalizing is 1.
    """

    def __init__(self, NiMC, batch_size):
        self.NiMC = NiMC
        self.Theta_index_non_const_dims = np.where(NiMC.Theta[:,1] - NiMC.Theta[:,0]>0)[0]

        def reward_function(initial_state, fidel):
            initial_state = self.get_full_state(initial_state).tolist()
            reward = NiMC.simulate(initial_state, fidel)
            return reward

        self.reward_function = reward_function

        self.domain_bounds = NiMC.Theta[self.Theta_index_non_const_dims, :]
        self.domain_dim = len(self.Theta_index_non_const_dims)

        if hasattr(NiMC, 'fidel_cost_function'):
            fidel_cost_function = NiMC.fidel_cost_function
        else:
            fidel_cost_function = lambda z: 1

        self.fidel_cost_function = fidel_cost_function
        self.fidel_bounds = np.array([(1,NiMC.k)])
        self.fidel_dim = 1
        self.opt_fidel_cost = self.cost_single(1)
        self.max_iteration = 200
        self.batch_size = batch_size

    # ---------------------------

    def get_full_state(self, state):
        _state = state.reshape(-1)
        state = self.NiMC.Theta[:,0].copy()
        if len(_state) == len(self.Theta_index_non_const_dims):
            state[self.Theta_index_non_const_dims] = _state
        elif len(_state) == len(state):
            state[:] = _state
        else:
            raise ValueError('Wrong size')
        return state

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
