import numpy as np
from MFTreeSearchCV.MFHOO import *
from pympler.asizeof import asizeof

# -----------------------------------------------------------------------------

useHOO = True

def estimate_max_probability(mfobject, num_HOO, rho_max, sigma, budget, debug=False):
    MP = MFPOO(mfobject=mfobject, nu_max=1.0, rho_max=rho_max, nHOO=num_HOO, sigma=sigma, C=0.1, mult=0.5, tol=1e-3,
               Randomize=False, Auto=True if not useHOO else False, unit_cost=mfobject.opt_fidel_cost, useHOO=useHOO, direct_budget=budget)
    MP.run_all_MFHOO()
    X, Depth, Cells = MP.get_point()
    # print(MP.number_of_queries)

    init_state = np.zeros((1, mfobject.domain_dim))
    fidelity = mfobject.fidel_bounds[0][1]

    mean_values = []

    def evaluate_single_state(X, mult):
        for j in range(mfobject.domain_dim):
            init_state[0][j] = X[j]
        value = 0
        for i in range(mfobject.max_iteration * mult):
            reward = mfobject.reward_function(init_state, fidelity)
            value = value + reward
        value = value / (mfobject.max_iteration * mult)
        return value

    mult = 50

    for k in range(len(X)):
        value = evaluate_single_state(X[k], mult)
        mean_values = mean_values + [value]

    best_instance = np.argmax(mean_values)
    best_value = mean_values[best_instance]
    memory_usage = asizeof(MP)
    if debug:
        print('best Cells for each smoothness: ')
        print(str(Cells))
        print('----------------------------------------------------------------------')
        print('best states for each smoothness: ')
        print(str(X))
        print('----------------------------------------------------------------------')
        print('Depth for each smoothness: ' + str(Depth))
        print('----------------------------------------------------------------------')
        print('max probability of hitting US for each smoothness: ' + str(mean_values))
        print('best max probability: %lf, memory usage: %.3f MB'%(best_value, memory_usage/1024.0/1024.0))
        print('----------------------------------------------------------------------')

    return X[best_instance], best_value, np.max(Depth), memory_usage, sum(MP.number_of_queries)

# -----------------------------------------------------------------------------
