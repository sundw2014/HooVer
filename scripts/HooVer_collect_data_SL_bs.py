# Author: Dawei Sun
import sys
sys.path.append('..')
import numpy as np
import os
import importlib

from utils.general_utils import loadpklz, savepklz, evaluate_single_state
import MFMC

# -----------------------------------------------------------------------------

model = 'Slplatoon3'
exp_id = int(sys.argv[1])

if __name__ == '__main__':
    budget = 800000
    outputs = []
    bss = [10, 20, 50, 80, 100, 200, 400, 800]
    # run an experiment for each batch_size configuration
    for bs in bss[::-1]:
        filename = 'data/HooVer_%s_budget%d_bs%d_exp%d.pklz'%(model, budget, bs, exp_id)
        os.system('cd ../; python example.py --nRuns 1 --model %s --budget %d --batch_size %d --filename %s'%(model, budget, bs, filename))
        outputs.append(loadpklz('../'+filename))

    results = []

    optimal_xs = [o['optimal_xs'][0] for o in outputs]
    original_results = [o['optimal_values'][0] for o in outputs]
    num_nodes = [o['num_nodes'][0] for o in outputs]

    # Monte-Carlo estimation of the hitting probability (using 250k simulations)
    simulator = importlib.import_module('models.'+model)
    MFMC.set_simulator(simulator)
    _, mch = MFMC.get_mch_as_mf(batch_size = 1)
    for initial_states in optimal_xs:
        initial_states = initial_states.tolist()
        np.random.seed(1024)
        result = evaluate_single_state(mch.run_markov_chain, initial_states, simulator.T, mult=250000)
        results.append(result)
        print(result)

    savepklz({'results':results, 'num_nodes': num_nodes, 'bss':bss, 'original_results':original_results}, '../data/HooVer_%s_bs_exp%d.pklz'%(model, exp_id))
