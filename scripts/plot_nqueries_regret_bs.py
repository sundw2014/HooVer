# Author: Rajat Sen # Modified by Negin
import sys
sys.path.append('..')
import numpy as np
import subprocess
from subprocess import DEVNULL, STDOUT, check_call
import os, signal

from utils.general_utils import loadpklz, savepklz

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

useHOO = True
model = 'Slplatoon3'
T = 11
dim = 4
port_base = 9100
plasmalab_root = '/home/daweis2/plasmalab-1.4.4/'

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
    plt.show()
