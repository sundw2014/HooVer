# import sys
# sys.path.append('..')
import numpy as np
from MFTreeSearchCV.MFHOO import *
from pympler.asizeof import asizeof

def estimate_max_probability(mfobject, nu, rho, sigma, C, t0, direct_budget, useHOO, verbose=False):
    budget = 5 * mfobject.opt_fidel_cost
    MP = MFPOO(mfobject=mfobject, nu_max=nu, rho_max=rho, total_budget=budget, sigma=sigma, C=C, mult=0.5, tol=1e-3,
               Randomize=False, Auto=True if not useHOO else False, unit_cost=t0, useHOO=useHOO, direct_budget=direct_budget)
    MP.run_all_MFHOO()
    X, Depth, Cells = MP.get_point()
    if verbose: print(MP.t)

    total_number_of_queries = np.sum(MP.number_of_queries)

    run_markov_chain = mfobject.reward_function

    init_state = np.zeros((1, mfobject.domain_dim))
    fidelity = mfobject.fidel_bounds[0][1] # use full fidelity when evaluating the probability

    mean_values = []

    def evaluate_single_state(X, mult):
        for j in range(mfobject.domain_dim):
            init_state[0][j] = X[j]
        value = 0
        # np.random.seed(1024)
        for i in range(mfobject.max_iteration * mult):
            reward = run_markov_chain(init_state, fidelity)
            value = value + reward
        value = value / (mfobject.max_iteration * mult)
        return value

    mult = 10

    for k in range(len(X)):
        value = evaluate_single_state(X[k], mult)
        mean_values = mean_values + [value]

    best_instance = np.argmax(mean_values)
    mult = 100
    best_value = evaluate_single_state(X[best_instance], mult)
    memory_usage = asizeof(MP)
    if verbose:
        print('best Cells for each smoothness: ')
        print(str(Cells))
        print('----------------------------------------------------------------------')
        print('best states for each smoothness: ')
        print(str(X))
        print('----------------------------------------------------------------------')
        print('Depth for each smoothness: ' + str(Depth))
        print('----------------------------------------------------------------------')
        print('max probability: %lf, memory usage: %.3f MB'%(best_value, memory_usage/1024.0/1024.0))
        print('----------------------------------------------------------------------')

    return total_number_of_queries, best_value, np.max(Depth), memory_usage
