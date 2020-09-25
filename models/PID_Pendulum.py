#!/usr/bin/env python
import numpy as np
import sys
sys.path.append('..')
from NiMC import NiMC
from simple_pid import PID

__all__ = ['PID_Pendulum']

m = 0.15
g = 9.81
l = 0.5
dt = 0.03

def dynamics(x, u):
    theta, dtheta = list(x)
    theta_next = theta + dt * dtheta
    dtheta_next = dtheta + dt * ((g / l) * np.sin(theta) - (0.1 / (m*(l**2))) * dtheta + 1 / (m * (l**2)) * u)
    return [theta_next, dtheta_next]

class PID_Pendulum(NiMC):
    def __init__(self, k=0):
        super(PID_Pendulum, self).__init__()
        self.set_Theta([[3,6], [0., 0.02], [0, 0.2]])
        self.set_k(k)

    def is_unsafe(self, state):
        # put every thing here
        pid = PID(state[0], state[1], state[2], setpoint=0.)
        def _is_unsafe(state):
            if state[0] > np.pi/2 or state[0] < -np.pi/2:
                return True
            return False

        X_MIN = np.array([-np.pi/3, -np.pi/3])
        X_MAX = np.array([np.pi/3, np.pi/3])
        Xs = []
        X = (np.random.rand(2) * (X_MAX - X_MIN) + X_MIN).tolist()
        Xs.append(X)
        T = 100
        for t in range(T):
            print(X[0])
            u = pid(X[0])
            X = dynamics(X, u)
            Xs.append(X)
            # if _is_unsafe(X):
            #     return True
        # return False
        return Xs

    def transition(self, state):
        assert len(state) == self.Theta.shape[0]
        return state

from matplotlib import pyplot as plt
model=PID_Pendulum()

for i in range(10): 
    Xs = model.is_unsafe([10., 0., 0.]) 
    plt.plot([X[0] for X in Xs])
    plt.show()
