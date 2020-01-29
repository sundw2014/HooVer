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
plasmalab_root = '/root/plasmalab-1.4.4/'

import models.FakeModel as simulator

if __name__ == '__main__':
    budget = 400000
    results = []
    original_results = []
    ss = np.array([1.7, 2.6, 2.9, 3.2, 3.6]) * 1e-4
    print(ss)
    for s in ss:
        filename = 'data/HooVer_%s_budget%d_s%lf_exp%d.pklz'%(model, budget, s, exp_id)
        os.system('cd ../; python example.py --nRuns 1 --sigma 1e-5 --model %s_%lf --budget %d --filename %s'%(model, s, budget, filename))
        simulator.s = s
        initial_states = loadpklz('../'+filename)['optimal_xs'][0]
        original_results.append(loadpklz('../'+filename)['optimal_values'][0])
        result = simulator.get_prob(initial_states)
        results.append(result)

    savepklz({'results':results, 'ss':ss, 'original_results':original_results}, '../data/HooVer_%s_exp%d.pklz'%(model, exp_id))
    # from IPython import embed; embed()
