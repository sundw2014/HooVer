import sys
sys.path.append('..')
import numpy as np
from utils.general_utils import loadpklz, savepklz

# -----------------------------------------------------------------------------

model = 'Slplatoon'

if __name__ == '__main__':
    budget = 800000
    outputs = []
    bss = [10, 20, 50, 100, 400, 1600, 6400, 25600]
    for exp_id in range(1,10):
        outputs.append([])
        for bs in bss:
            filename = 'data/HooVer_%s_budget%d_bs%d_exp%d.pklz'%(model, budget, bs, exp_id)
            outputs[-1].append(loadpklz('../'+filename))

    depth = [[o['depths'][0] for o in exp] for exp in outputs]
    mem = [[o['memory_usages'][0] for o in exp] for exp in outputs]
    time = [[o['running_times'][0] for o in exp] for exp in outputs]
    num_nodes = [[o['num_nodes'][0] for o in exp] for exp in outputs]

    depth = np.array(depth).mean(axis=0)
    mem = np.array(mem).mean(axis=0)
    time = np.array(time).mean(axis=0)
    num_nodes = np.array(num_nodes).mean(axis=0)

    results_ho = []
    for exp_id in range(1, 10):
        results_ho.append(loadpklz('../data/HooVer_%s_bs_exp%d.pklz'%(model, exp_id))['results'][::-1])

    results_ho = np.array(results_ho)

print('\\hline')
print('{\\bf b\_size} & '+(' & '.join(['%d' for _ in range(len(bss))]))%tuple(bss)+'\\\\')
print('\\hline')
print('{\\bf \#Nodes} & '+(' & '.join(['%d' for _ in range(len(bss))]))%tuple(num_nodes)+'\\\\')
print('\\hline')
print('{\\bf Running Time\/} (s) & '+(' & '.join(['%d' for _ in range(len(bss))]))%tuple(time)+'\\\\')
print('\\hline')
print('{\\bf Memory (Mb)\/} & '+(' & '.join(['%.2f' for _ in range(len(bss))]))%tuple(mem)+'\\\\')
print('\\hline')
print('{\\bf Result\/} & '+(' & '.join(['%.4f' for _ in range(len(bss))]))%tuple(results_ho.mean(axis=0))+'\\\\')
print('\\hline')
