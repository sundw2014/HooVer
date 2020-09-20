#!/usr/bin/env python
import numpy as np
import sys
sys.path.append('..')
from NiMC import NiMC
from simple_pid import PID
import math, time, sys
from scipy.linalg import expm

__all__ = ['LinearOutputFeedback']

class LinearOutputFeedback(NiMC):
    def __init__(self, k=0, n=2, m=2, l=2, A=None, B=None, C=None):
        super(LinearOutputFeedback, self).__init__()
        self.set_Theta([[-1., 1.] for _ in range(m*n)])
        self.set_k(k)
        self.n = n
        self.m = m
        self.A = A if A is not None else np.random.randn(self.n, self.n)
        self.B = B if B is not None else np.random.randn(self.n, self.m)
        if C is not None:
            self.C = C
        elif self.n == self.m:
            self.C = np.eye(self.n)
        else:
            self.C = np.random.randn(self.l, self.n)

    def _simulate(self, K):
        K = np.array(K).reshape(self.m, self.n)
        A_cl = self.A - self.B.dot(K)
        X_MIN = np.array([-1. for _ in range(self.n)])
        X_MAX = np.array([ 1. for _ in range(self.n)])
        x = np.random.rand() * (X_MAX - X_MIN) + X_MIN
        T = 1.
        return -np.sqrt((expm(A_cl*T).dot(x.reshape([-1,1]))**2).sum() / ((x**2).sum()))

    def simulate(self, initial_state, k=None):
        # increases by 1 every time the simulate function gets called
        self.cnt_queries += 1

        if k is None:
            k = self.k

        state = initial_state
        return self._simulate(state)

    def is_unsafe(self, state):
        # put every thing here
        raise NotImplementedError('is_unsafe')

    def transition(self, state):
        assert len(state) == self.Theta.shape[0]
        return state
