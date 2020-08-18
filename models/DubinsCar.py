import numpy as np
import sys
sys.path.append('..')
from NiMC import NiMC

__all__ = ['DubinsCar']

# some constants
state_start = np.array([0,-0.5,0])
state_range = np.array([0,1,0])

Theta = np.stack([state_start, state_start + state_range]).T
dt = 0.1
k = 50

class DubinsCar(NiMC):
    def __init__(self, k=k):
        super(DubinsCar, self).__init__()
        self.set_Theta(Theta)
        self.set_k(k)

    def is_unsafe(self, state):
        if state[1] > 1 or state[1] < -1:
            return True
        return False

    def transition(self, state):
        assert len(state) == 3
        x,y,theta = state
        v, omega = np.array([1, 0]) + np.array([0.3, 0.3]) * np.random.randn(2)
        state = np.array(state) + dt * np.array([v*np.cos(theta), v*np.sin(theta), omega])
        return state
