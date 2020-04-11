import numpy as np
from MFTreeSearchCV.MFHOO import *
from pympler.asizeof import asizeof
from MFTreeSearchCV.MFNiMC import MFNiMC
# -----------------------------------------------------------------------------

useHOO = True

def estimate_max_probability(nimc, num_HOO, rho_max, sigma, budget, batch_size, debug=False):
    mfobject = MFNiMC(nimc, batch_size)
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
        for i in range(mult):
            reward = mfobject.reward_function(init_state, fidelity)
            value = value + reward
        value = value / (mult)
        return value

    mult = 10000

    for k in range(len(X)):
        value = evaluate_single_state(X[k], mult)
        mean_values = mean_values + [value]

    best_instance = np.argmax(mean_values)
    best_value = mean_values[best_instance]
    best_x = mfobject.get_full_state(X[best_instance])
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

    return best_x, best_value, np.max(Depth), memory_usage, sum(MP.number_of_queries)

# -----------------------------------------------------------------------------
