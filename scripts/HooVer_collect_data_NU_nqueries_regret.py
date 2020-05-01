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
    ss = np.array([1e-1, 1e-2, 1e-3, 3e-4])
    _budgets = [[0.412, 1.0], [0.412, 1.0], [0.5, 1.5], [1.0, 3.0]]
    _budgets = [np.logspace(np.log(start * 1e5)/np.log(2), np.log(end * 1e5)/np.log(2), num=20, base=2).astype('int') for start,end in _budgets]

    results = []
    original_results = []
    print(ss)
    for s, budgets in zip(ss, _budgets):
        s_results = []
        for budget in budgets:
            filename = 'data/HooVer_%s_nqueries_regret_budget%d_s%lf_exp%d.pklz'%(model, budget, s, exp_id)
            os.system('cd ../; python3 check.py --nRuns 1 --sigma 1e-5 --model %s --args %lf --budget %d --output %s --seed %d'%(model, s, budget, filename, exp_id*1024))
            nimc = models.__dict__[model](s=s)
            initial_states = loadpklz('../'+filename)['optimal_xs'][0]
            original_results.append(loadpklz('../'+filename)['optimal_values'][0])
            result = nimc.get_prob(initial_states)
            s_results.append(result)
        results.append(s_results)
    savepklz({'results':results, 'ss':ss, 'budgets':_budgets, 'original_results':original_results}, '../data/HooVer_%s_nqueries_regret_exp%d.pklz'%(model, exp_id))
    # from IPython import embed; embed()
