import numpy as np
import sys
sys.path.append('..')
from NiMC import NiMC

__all__ = ['MyModel']

class MyModel(NiMC):
    def __init__(self, sigma=0.1, k=10):
        super(MyModel, self).__init__()
        self.set_Theta([[1,2],[2,3]])
        self.set_k(k)
        self.sigma = sigma

    def is_unsafe(self, state):
        if np.linalg.norm(state) > 4:
            return True # return unsafe if norm of the state is greater than 4.
        return False # return safe otherwise.

    def transition(self, state):
        state = np.array(state)
        # increment is a 2-dimensional
        # normally distributed vector
        increment = self.sigma * np.random.randn(2)
        state += increment
        state = state.tolist()
        return state # return the new state
