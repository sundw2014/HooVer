import numpy as np
import sys
sys.path.append('..')
from NiMC import NiMC

__all__ = ['FakeModel']

class FakeModel(NiMC):
    def __init__(self, s):
        super(FakeModel, self).__init__()
        self.s = s
        self.set_Theta([[0,1],[0,1]])
        self.set_k(0)

    def is_unsafe(self, state):
        prob = self.get_prob(state)
        return np.random.choice([False, True], p=[1-prob, prob])

    def get_prob(self, state):
        center = self.Theta.mean(dim=1)
        r = np.sqrt(np.sum((state-center)**2))
        prob = min(1.0, np.exp(-1.0 * r ** 2 / self.s))
        return prob

    def transition(self, state):
        assert len(state) == self.Theta.shape[0]
        return state
