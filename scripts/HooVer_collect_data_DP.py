import sys
sys.path.append('..')
import numpy as np
import os

from utils.general_utils import loadpklz, savepklz, evaluate_single_state
import models

# -----------------------------------------------------------------------------

model = 'DetectingPedestrian'
exp_id = int(sys.argv[1])

if __name__ == '__main__':
    budgets = np.logspace(np.log(0.65 * 1e5)/np.log(2), np.log(8e5)/np.log(2), num=7, base=2).astype('int')
    # plasmalab cuts down the budget somehow. Remove the last budget to make it consistent with the results of PlasmaLab.
    budgets = budgets[0:-1]
    outputs = []
    print('budgets: ' + str(budgets))
    # run an experiment for each budget configuration
    for budget in budgets:
        filename = 'data/HooVer_%s_budget%d_exp%d.pklz'%(model, budget, exp_id)
        os.system('cd ../; python3 check.py --nRuns 1 --model %s --budget %d --output %s --seed %d'%(model, budget, filename, exp_id*1024))
        outputs.append(loadpklz('../'+filename))

    results = []

    optimal_xs = [o['optimal_xs'][0] for o in outputs]
    original_results = [o['optimal_values'][0] for o in outputs]
    num_queries = [o['budget'] for o in outputs]

    # Monte-Carlo estimation of the hitting probability (using 250k simulations)
    nimc = models.__dict__[model]()
    for initial_states in optimal_xs:
        initial_states = initial_states.tolist()
        np.random.seed(1024)
        result = evaluate_single_state(nimc, initial_states, nimc.k, mult=250000)
        results.append(result)
        print(result)

    savepklz({'results':results, 'num_queries':num_queries, 'original_results':original_results}, '../data/HooVer_%s_exp%d.pklz'%(model, exp_id))
