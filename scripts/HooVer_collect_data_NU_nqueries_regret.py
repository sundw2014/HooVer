# Author: Rajat Sen # Modified by Negin
import sys
sys.path.append('..')
import numpy as np
import subprocess
from subprocess import DEVNULL, STDOUT, check_call
import os, signal

from utils.general_utils import loadpklz, savepklz

# -----------------------------------------------------------------------------

useHOO = True
model = 'FakeModel'
T = 1
dim = 2
exp_id = int(sys.argv[1])
port_base = 9100
plasmalab_root = '/home/daweis2/plasmalab-1.4.4/'

import models.FakeModel as simulator

if __name__ == '__main__':
    ss = np.array([1e-1, 1e-2, 1e-3, 3e-4])
    _budgets = [[0.408, 1.0], [0.408, 1.0], [0.5, 1.5], [1.0, 3.0]]
    _budgets = [np.logspace(np.log(start * 1e5)/np.log(2), np.log(end * 1e5)/np.log(2), num=20, base=2).astype('int') for start,end in _budgets]

    results = []
    original_results = []
    # ss = np.array([1.7, 2.6, 2.9, 3.2, 3.6]) * 1e-4
    # ss = np.array([3.7, 4.0]) * 1e-4
    # ss = np.array([1.7,]) * 1e-4
    print(ss)
    for s, budgets in zip(ss, _budgets):
        s_results = []
        for budget in budgets:
            filename = 'data/HooVer_%s_nqueries_regret_budget%d_s%lf_exp%d.pklz'%(model, budget, s, exp_id)
            os.system('cd ../; python example.py --nRuns 1 --sigma 1e-5 --model %s_%lf --budget %d --filename %s'%(model, s, budget, filename))
            simulator.s = s
            initial_states = loadpklz('../'+filename)['optimal_xs'][0]
            original_results.append(loadpklz('../'+filename)['optimal_values'][0])
            result = simulator.get_prob(initial_states)
            s_results.append(result)
        results.append(s_results)
    savepklz({'results':results, 'ss':ss, 'budgets':_budgets, 'original_results':original_results}, '../data/HooVer_%s_nqueries_regret_exp%d.pklz'%(model, exp_id))
    # from IPython import embed; embed()
