import sys
sys.path.append('..')
import numpy as np
import os

from utils.general_utils import loadpklz, savepklz, evaluate_single_state
import models

# -----------------------------------------------------------------------------

model = 'Slplatoon'
exp_id = int(sys.argv[1])

if __name__ == '__main__':
    budget = 800000
    outputs = []
    bss = [10, 20, 50, 100, 400, 1600, 6400, 25600]
    # run an experiment for each batch_size configuration
    for bs in bss[::-1]:
        filename = 'data/HooVer_%s_budget%d_bs%d_exp%d.pklz'%(model, budget, bs, exp_id)
        os.system('cd ../; python3 check.py --nRuns 1 --model %s --budget %d --batch_size %d --output %s --seed %d'%(model, budget, bs, filename, exp_id*1024))
        outputs.append(loadpklz('../'+filename))

    results = []

    optimal_xs = [o['optimal_xs'][0] for o in outputs]
    original_results = [o['optimal_values'][0] for o in outputs]
    num_nodes = [o['num_nodes'][0] for o in outputs]

    # Monte-Carlo estimation of the hitting probability (using 250k simulations)
    nimc = models.__dict__[model]()
    for initial_states in optimal_xs:
        initial_states = initial_states.tolist()
        np.random.seed(1024)
        result = evaluate_single_state(nimc, initial_states, nimc.k, mult=250000)
        results.append(result)
        print(result)

    savepklz({'results':results, 'num_nodes': num_nodes, 'bss':bss, 'original_results':original_results}, '../data/HooVer_%s_bs_exp%d.pklz'%(model, exp_id))
