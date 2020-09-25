#!/usr/bin/env python
import numpy as np
import sys
sys.path.append('..')
from NiMC import NiMC
import math, time, sys
import scipy
from scipy.linalg import expm
from scipy.integrate import odeint

np.random.seed(100)

def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    """
    # http://www.mwm.im/lqr-controllers-with-python/
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)
    print(np.allclose(A.T.dot(X) + X.dot(A) - X.dot(B).dot(np.linalg.inv(R)).dot(B.T).dot(X), -Q))
    return np.asarray(K), np.asarray(X), np.asarray(eigVals)

__all__ = ['LQR']

class LQR(NiMC):
    def __init__(self, k=0, sigma=0., N = 1000, n=2, m=2, l=2, A=None, B=None, C=None, Q=None, R=None, dt=0.01):
        super(LQR, self).__init__()
        self.set_Theta([[-1., 1.] for _ in range(m*n)])
        self.set_k(k)
        self.sigma = sigma
        self.N = N
        self.n = n
        self.m = m
        self.A = A if A is not None else np.random.randn(self.n, self.n)
        self.B = B if B is not None else np.random.randn(self.n, self.m)
        if Q is not None:
            self.Q = Q
        else:
            self.Q = np.random.randn(self.n, self.n)
            self.Q = self.Q.dot(self.Q.T)
        if R is not None:
            self.R = R
        else:
            self.R = np.random.randn(self.m, self.m)
            self.R = self.R.dot(self.R.T)
        if C is not None:
            self.C = C
        elif self.n == self.m:
            self.C = np.eye(self.n)
        else:
            self.C = np.random.randn(self.l, self.n)
        self.dt = dt
        self.K_gt, _, _ = lqr(self.A, self.B, self.Q, self.R)
        print(self.K_gt.reshape(-1))

    def _simulate(self, K):
        K = np.array(K).reshape(self.n, self.m)
        x = np.array([1. for _ in range(self.n)]).reshape(-1,1)
        cost = 0.
        # x_l = odeint(cl_linear, X0, t, args=(u,))
        for i in range(self.N):
            w = self.sigma * np.random.randn(*x.shape)
            u = K.dot(x)
            dx = self.A.dot(x) - self.B.dot(u) + w
            x += dx * self.dt
            cost += x.T.dot(self.Q).dot(x) * self.dt
            if i is not self.N-1:
                cost += u.T.dot(self.R).dot(u) * self.dt
            # print(x.T.dot(self.Q).dot(x) * self.dt, u.T.dot(self.R).dot(u) * self.dt)
        return 1. - min(cost * 0.1, 1.)

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
