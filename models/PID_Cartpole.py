# Author : Nikhil Advani
# Date : 22nd May 2017
# Description:  This programs runs a simulation of a cart-pole system
#               The conrol algorithm used to balance the pendulum can either be using LQR or PID
#               Please have a look at the README and the project report.
#!/usr/bin/env python
import numpy as np
import sys
sys.path.append('..')
from NiMC import NiMC
from simple_pid import PID
import math, time, sys

__all__ = ['PID_Cartpole']


class Cart:
    def __init__(self,x,mass,world_size):
        self.x = x  
        self.y = int(0.6*world_size)        # 0.6 was chosen for aesthetic reasons.
        self.mass = mass
        self.color = (0,255,0)

class Pendulum:
    def __init__(self,length,theta,ball_mass):
        self.length = length
        self.theta = theta
        self.ball_mass = ball_mass      
        self.color = (0,0,255)

def apply_control_input(cart,pendulum,F,time_delta,x_tminus2,theta_dot,theta_tminus2,previous_time_delta,g):
    # Finding x and theta on considering the control inputs and the dynamics of the system
    theta_double_dot = (((cart.mass + pendulum.ball_mass) * g * math.sin(pendulum.theta)) + (F * math.cos(pendulum.theta)) - (pendulum.ball_mass * ((theta_dot)**2.0) * pendulum.length * math.sin(pendulum.theta) * math.cos(pendulum.theta))) / (pendulum.length * (cart.mass + (pendulum.ball_mass * (math.sin(pendulum.theta)**2.0)))) 
    x_double_dot = ((pendulum.ball_mass * g * math.sin(pendulum.theta) * math.cos(pendulum.theta)) - (pendulum.ball_mass * pendulum.length * math.sin(pendulum.theta) * (theta_dot**2)) + (F)) / (cart.mass + (pendulum.ball_mass * (math.sin(pendulum.theta)**2)))
    cart.x += ((time_delta**2) * x_double_dot) + (((cart.x - x_tminus2) * time_delta) / previous_time_delta)
    pendulum.theta += ((time_delta**2)*theta_double_dot) + (((pendulum.theta - theta_tminus2)*time_delta)/previous_time_delta)

def find_error(pendulum):
    # There's a seperate function for this because of the wrap-around problem
    # This function returns the error
    previous_error = (pendulum.theta % (2 * math.pi)) - 0
    if previous_error > math.pi:
        previous_error = previous_error - (2 * math.pi)
    return previous_error

def simulate(Kp, Kd, Ki):
    Kp = Kp * 100
    Kd = Kd * 10
    Ki = Ki * 10
    def find_pid_control_input(cart,pendulum,time_delta,error,previous_error,integral,g):
        # Using PID to find control inputs
        derivative = (error - previous_error) / time_delta
        integral += error * time_delta
        F = (Kp * error) + (Kd * derivative) + (Ki * integral)
        return F,integral

    # Initializing mass values, g, world size, simulation time and variables required to terminate the simulation
    mass_of_ball = 1.0
    mass_of_cart = 5.0
    g = 9.81
    errors, force, theta, times, x = [],[],[],[],[]
    world_size = 1000
    simulation_time = 35

    previous_timestamp = time.time()
    end_time = previous_timestamp + simulation_time

    # Initializing cart and pendulum objects
    cart = Cart(int(0.2 * world_size),mass_of_cart,world_size)

    X_MIN = -np.pi/3
    X_MAX = np.pi/3
    X = np.random.rand() * (X_MAX - X_MIN) + X_MIN

    pendulum = Pendulum(1,X,mass_of_ball)


    # Initializing other variables needed for the simulation
    theta_dot = 0
    theta_tminus1 = theta_tminus2 = pendulum.theta
    x_tminus1 = x_tminus2 = cart.x
    previous_error = find_error(pendulum)
    integral = 0
    time_delta = 0.03
    previous_time_delta = time_delta

    def _is_unsafe(pendulum):
        if pendulum.theta > np.pi/2 or pendulum.theta < -np.pi/2:
            return True
        return False

    # The simulation must run for the desired amount of time
    T = 100
    for t in range(T):
        error = find_error(pendulum)
        if previous_time_delta != 0:    # This condition is to make sure that theta_dot is not infinity in the first step
            theta_dot = (theta_tminus1 - theta_tminus2 ) / previous_time_delta              
            x_dot = (x_tminus1 - x_tminus2) / previous_time_delta
            F,intergral = find_pid_control_input(cart,pendulum,time_delta,error,previous_error,integral,g)
            apply_control_input(cart,pendulum,F,time_delta,x_tminus2,theta_dot,theta_tminus2,previous_time_delta,g)
            
            # For plotting the graphs
            force.append(F)
            x.append(cart.x)
            errors.append(error)        
            theta.append(pendulum.theta)
    
        # Update the variables and display stuff
        previous_time_delta = time_delta
        previous_error = error
        theta_tminus2 = theta_tminus1
        theta_tminus1 = pendulum.theta
        x_tminus2 = x_tminus1
        x_tminus1 = cart.x
    return 1. - (np.array(errors) / errors[0]).mean()

# from matplotlib import pyplot as plt
# for i in range(10):
#     errors = simulate(-1.5, -2, -2)
#     print(errors)
#     plt.plot(errors)
#     plt.show()

class PID_Cartpole(NiMC):
    def __init__(self, k=0):
        super(PID_Cartpole, self).__init__()
        self.set_Theta([[-2.0, -0.5], [-3., -1.], [-3., -1.]])
        self.set_k(k)

    def simulate(self, initial_state, k=None):
        # increases by 1 every time the simulate function gets called
        self.cnt_queries += 1

        if k is None:
            k = self.k

        state = initial_state
        return simulate(state[0], state[1], state[2])

    def is_unsafe(self, state):
        # put every thing here
        raise NotImplementedError('is_unsafe')

    def transition(self, state):
        assert len(state) == self.Theta.shape[0]
        return state

# from matplotlib import pyplot as plt
# model=PID_Pendulum()

# for i in range(10): 
#     Xs = model.is_unsafe([10., 0., 0.]) 
#     plt.plot([X[0] for X in Xs])
#     plt.show()
