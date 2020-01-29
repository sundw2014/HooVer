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
    num_nodes = []
    bss = [10, 20, 50, 80, 100, 200, 400, 800]
    for exp_id in range(1,10):
        outputs.append([])
        num_nodes.append([])
        for bs in bss[::-1]:
            filename = 'data/HooVer_%s_budget%d_bs%d_exp%d.pklz'%(model, budget, bs, exp_id)
            outputs[-1].append(loadpklz('../'+filename))
            num_nodes[-1].append(budget/bs)

    depth = [[o['depths'][0] for o in exp] for exp in outputs]
    mem = [[o['memory_usages'][0] for o in exp] for exp in outputs]
    time = [[o['running_times'][0] for o in exp] for exp in outputs]

    depth = np.array(depth).mean(axis=0)
    mem = np.array(mem).mean(axis=0)
    time = np.array(time).mean(axis=0)
    num_nodes = np.array(num_nodes).mean(axis=0)

    results_ho = []
    for exp_id in range(1, 10):
        results_ho.append(loadpklz('../data/HooVer_%s_bs_exp%d.pklz'%(model, exp_id))['results'])

    results_ho = np.array(results_ho)

print('batch_size: '+' '.join([str(s) for s in bss]))
print('#nodes: '+' '.join([str(s) for s in num_nodes]))
print('MEM: '+' '.join([str(s) for s in mem]))
print('TIME: '+' '.join([str(s) for s in time]))
print('Result: '+' '.join([str(s) for s in results_ho.mean(axis=0)]))
# from IPython import embed;
# embed()
