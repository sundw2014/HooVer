# Author: Rajat Sen # Modified by Negin
import sys
sys.path.append('..')
import numpy as np
from utils.general_utils import loadpklz, savepklz

# -----------------------------------------------------------------------------

model = 'Slplatoon3'

if __name__ == '__main__':
    budget = 800000
    outputs = []
    rhomaxs = [0.95, 0.9, 0.8, 0.6, 0.4, 0.16, 0.01]
    for exp_id in range(1,10):
        outputs.append([])
        for bs in bss[::-1]:
            filename = 'data/HooVer_%s_budget%d_bs%d_exp%d.pklz'%(model, budget, bs, exp_id)
            outputs[-1].append(loadpklz('../'+filename))

    depth = [[o['depths'][0] for o in exp] for exp in outputs]

    depth = np.array(depth).mean(axis=0)

    results_ho = []
    for exp_id in range(1, 10):
        results_ho.append(loadpklz('../data/HooVer_%s_rhomax_exp%d.pklz'%(model, exp_id))['results'])

    results_ho = np.array(results_ho)

print('rhomax: '+' '.join([str(s) for s in rhomaxs]))
print('Depth: '+' '.join([str(s) for s in depth]))
print('Result: '+' '.join([str(s) for s in results_ho.mean(axis=0)]))
# from IPython import embed;
# embed()
