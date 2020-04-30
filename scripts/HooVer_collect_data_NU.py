import sys
sys.path.append('..')
import numpy as np
import os

from utils.general_utils import loadpklz, savepklz, evaluate_single_state
import models

# -----------------------------------------------------------------------------

model = 'ConceptualModel'
exp_id = int(sys.argv[1])

if __name__ == '__main__':
    budget = 400000
    results = []
    original_results = []
    ss = np.array([1.7, 2.6, 2.9, 3.2, 3.6]) * 1e-4
    print(ss)
    for s in ss:
        filename = 'data/HooVer_%s_budget%d_s%lf_exp%d.pklz'%(model, budget, s, exp_id)
        os.system('cd ../; python3 check.py --nRuns 1 --sigma 1e-5 --model %s --args %lf --budget %d --output %s --seed %d'%(model, s, budget, filename, exp_id*1024))
        nimc = models.__dict__[model]()
        initial_states = loadpklz('../'+filename)['optimal_xs'][0]
        original_results.append(loadpklz('../'+filename)['optimal_values'][0])
        result = nimc.get_prob(initial_states)
        results.append(result)

    savepklz({'results':results, 'ss':ss, 'original_results':original_results}, '../data/HooVer_%s_exp%d.pklz'%(model, exp_id))
    # from IPython import embed; embed()
