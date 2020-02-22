import sys
sys.path.append('..')
from utils.general_utils import loadpklz, savepklz
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

model = 'FakeModel'

if __name__ == '__main__':
    exps_ho = [loadpklz('../data/HooVer_%s_nqueries_regret_exp%d.pklz'%(model, exp_id)) for exp_id in range(1, 11)]
    ss = exps_ho[0]['ss']
    budgets_ho = exps_ho[0]['budgets']
    # from IPython import embed; embed()
    results_ho = [np.stack([exp['results'][ids] for exp in exps_ho]).mean(axis=0) for ids in range(len(ss))]

    exps_pl = [loadpklz('../data/PlasmaLab_%s_nqueries_regret_exp%d.pklz'%(model, exp_id)) for exp_id in range(1, 11)]
    ss = exps_pl[0]['ss']
    budgets_pl = exps_pl[0]['budgets']
    results_pl = [np.stack([exp['results'][ids] for exp in exps_pl]).mean(axis=0) for ids in range(len(ss))]

    colors = ['r','g','b','k','y']
    smooth_sigma = 1
    smooth_func = lambda x: scipy.ndimage.filters.gaussian_filter1d(x, sigma=smooth_sigma)

    for ids, s in enumerate(ss):
        result = smooth_func(1-results_ho[ids])
        plt.plot(budgets_ho[ids]/1e5, result, '-', label='s=%.2fe-2'%(s*1e2), color=colors[ids])
        # plt.text(budgets_ho[ids][len(budgets_ho[ids])//2]/1e5, result[len(budgets_ho[ids])//2], 's=%.2fe-2'%(s*1e2))

        result = smooth_func(1-results_pl[ids])
        plt.plot(budgets_pl[ids]/1e5*16, result, '--', color=colors[ids])
        # plt.text(budgets_pl[ids][len(budgets_pl[ids])//2]/1e5*16, result[len(budgets_pl[ids])//2], 's=%.2fe-2'%(s*1e2))


    plt.xlabel('#queries (x $10^5$)')
    plt.ylabel('regret')
    plt.legend()
    plt.show()
