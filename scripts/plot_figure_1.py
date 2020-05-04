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
top = 0.93     # the top of the subplots of the figure

models = ['Slplatoon', 'Mlplatoon18', 'DetectingPedestrian', 'Merging', 'Mlplatoon']
# models = ['Merging']
titles = ['$\mathsf{SLplatoon}(d=4, k=11)$','$\mathsf{MLplatoon}(d=8, k=9)$','$\mathsf{DetectBrake}(d=4 , k=10)$','$\mathsf{Merging}(d=4, k=10)$', '$\mathsf{MLplatoon}(d=9, k=9)$']
models = [models[int(sys.argv[1])-1], ]

for model in models:
    results_pl = []
    num_queries_pl = []
    results_ho = []
    num_queries_ho = []
    for exp_id in range(1, 10):
        results_pl.append(loadpklz('../data/PlasmaLab_%s_exp%d.pklz'%(model, exp_id))['results'])
        num_queries_pl.append(loadpklz('../data/PlasmaLab_%s_exp%d.pklz'%(model, exp_id))['num_queries'])
        results_ho.append(loadpklz('../data/HooVer_%s_exp%d.pklz'%(model, exp_id))['results'])
        num_queries_ho.append(loadpklz('../data/HooVer_%s_exp%d.pklz'%(model, exp_id))['num_queries'])

    results_pl = np.array(results_pl)
    num_queries_pl = np.array(num_queries_pl) / 1e5
    results_ho = np.array(results_ho)
    num_queries_ho = np.array(num_queries_ho) / 1e5

    plt.errorbar(num_queries_pl.mean(axis=0), results_pl.mean(axis=0), results_pl.std(axis=0), capsize=5.0, fmt='-ok', label='PlasmaLab')
    plt.errorbar(num_queries_ho.mean(axis=0), results_ho.mean(axis=0), results_ho.std(axis=0), capsize=5.0, fmt='-or', label='HooVer')
    plt.legend(loc='lower right')
    plt.xlabel('#queries (x $10^5$)')
    plt.ylabel('hitting probability')
    plt.title(titles[int(sys.argv[1])-1])
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    # plt.show()
    plt.savefig('../results/%s.pdf'%model)
