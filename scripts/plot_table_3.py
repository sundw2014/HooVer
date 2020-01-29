import sys
sys.path.append('..')
from utils.general_utils import loadpklz, savepklz
import numpy as np

results_pl = []
ss_pl = []
results_ho = []
ss_ho = []

model = 'FakeModel'

for exp_id in range(1, 10):
    results_pl.append(loadpklz('../data/PlasmaLab_%s_exp%d.pklz'%(model, exp_id))['results'])
    ss_pl.append(loadpklz('../data/PlasmaLab_%s_exp%d.pklz'%(model, exp_id))['ss'])

    if exp_id == 5:
        exp_id = 15
    results_ho.append(loadpklz('../data/HooVer_%s_exp%d.pklz'%(model, exp_id))['results'])
    ss_ho.append(loadpklz('../data/HooVer_%s_exp%d.pklz'%(model, exp_id))['ss'])

results_pl = np.array(results_pl)
results_ho = np.array(results_ho)
print('s: '+' '.join([str(s) for s in ss_ho[0]]))
print('PlasmaLab: '+' '.join([str(s) for s in results_pl.mean(axis=0)]))
print('HooVer: '+' '.join([str(s) for s in results_ho.mean(axis=0)]))
# from IPython import embed;
# embed()
