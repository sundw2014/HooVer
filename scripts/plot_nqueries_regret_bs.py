import sys
sys.path.append('..')
from utils.general_utils import loadpklz, savepklz
import numpy as np
import matplotlib.pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 13
HUGE_SIZE = 25

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=HUGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=HUGE_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)

left = 0.17  # the left side of the subplots of the figure
right = 0.98   # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.98     # the top of the subplots of the figure

# -----------------------------------------------------------------------------

model = 'Slplatoon'

if __name__ == '__main__':
    budgets = np.logspace(np.log(0.85 * 1e5)/np.log(2), np.log(8e5)/np.log(2), num=6, base=2).astype('int')
    outputs = []
    num_nodes = []
    # bss = [1, 10, 20, 50, 80, 100, 200, 400, 800]
    bss = [10, 20, 50, 80, 100, 200, 400, 800]
    exps = [loadpklz('../data/HooVer_%s_nqueries_regret_bs_exp%d.pklz'%(model, exp_id)) for exp_id in range(1, 11)]
    data = np.zeros((len(bss), len(budgets), len(exps)))

    budgets = budgets / 1e5
    for budget in range(len(budgets)):
        for bs in range(len(bss)):
            for exp_id in range(1, 11):
                data[bs, budget, exp_id-1] = exps[exp_id-1]['results'][budget*len(bss)+bs]

    for ibs, bs in enumerate(bss):
        plt.plot(budgets, 1-data.mean(axis=2)[ibs, :], 'o-', label='bs=%d'%bs)
        plt.text(budgets[0], 1-data.mean(axis=2)[ibs, :][0], 'bs=%d'%bs)
        plt.text(budgets[-1], 1-data.mean(axis=2)[ibs, :][-1], 'bs=%d'%bs)

    plt.xlabel('#queries (x $10^5$)')
    plt.ylabel('regret')
    plt.legend()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    plt.savefig('%s_bs_result_nqueries.pdf'%model)
