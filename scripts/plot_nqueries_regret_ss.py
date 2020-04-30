import sys
sys.path.append('..')
from utils.general_utils import loadpklz, savepklz
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

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

model = 'ConceptualModel'

if __name__ == '__main__':
    labels = ['s=0.1', 's=0.01', 's=0.001', 's=0.0003']
    exps_ho = [loadpklz('../data/HooVer_%s_nqueries_regret_exp%d.pklz'%(model, exp_id)) for exp_id in range(1, 11)]
    ss = exps_ho[0]['ss']
    budgets_ho = exps_ho[0]['budgets']
    results_ho = [np.stack([exp['results'][ids] for exp in exps_ho]).mean(axis=0) for ids in range(len(ss))]

    exps_pl = [loadpklz('../data/PlasmaLab_%s_nqueries_regret_exp%d.pklz'%(model, exp_id)) for exp_id in range(1, 11)]
    ss = exps_pl[0]['ss']
    budgets_pl = exps_pl[0]['budgets']
    results_pl = [np.stack([exp['results'][ids] for exp in exps_pl]).mean(axis=0) for ids in range(len(ss))]

    colors = ['r','g','b','k','y']
    smooth_sigma = 1
    smooth_func = lambda x: scipy.ndimage.filters.gaussian_filter1d(x, sigma=smooth_sigma)

    for ids, s in enumerate(ss):
        #result = smooth_func(1-results_ho[ids])
        result = smooth_func(results_ho[ids])
        plt.plot(budgets_ho[ids]/1e5, result, '-', label=labels[ids], color=colors[ids])
        # plt.text(budgets_ho[ids][len(budgets_ho[ids])//2]/1e5, result[len(budgets_ho[ids])//2], 's=%.2fe-2'%(s*1e2))

        #result = smooth_func(1-results_pl[ids])
        result = smooth_func(results_pl[ids])
        plt.plot(budgets_pl[ids]/1e5*16, result, '--', color=colors[ids])
        # plt.text(budgets_pl[ids][len(budgets_pl[ids])//2]/1e5*16, result[len(budgets_pl[ids])//2], 's=%.2fe-2'%(s*1e2))


    plt.xlabel('#queries (x $10^5$)')
    plt.ylabel('hitting probability')
    plt.legend()
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    plt.savefig('../results/%s_ss_result_nqueries.pdf'%model)
