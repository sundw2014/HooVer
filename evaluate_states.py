# import sys
# sys.path.append('..')
import sys
# sys.path.append('..')
import argparse
import numpy as np
import time
import importlib
import hoover
import MFMC
from utils.general_utils import loadpklz, savepklz

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from functools import partial

from multiprocessing import Pool, TimeoutError
import time
import os
import tqdm

simulator = importlib.import_module('models.DetectingPedestrian')

MFMC.set_simulator(simulator)

MCh_object, mch = MFMC.get_mch_as_mf(1)

def evaluate_single_state(init_state, budget, mult=10000):
    init_state = init_state.reshape(1,-1)
    true_max_prob = 0
    for i in range(mult):
        # import pdb; pdb.set_trace()
        reward = mch.run_markov_chain(init_state, budget)
        true_max_prob = true_max_prob + reward
    true_max_prob = true_max_prob / mult
    return true_max_prob

# initial_states = np.meshgrid(np.arange(0,5,0.1), np.arange(10,10.1,1), np.arange(20,20.1,1), np.arange(30,30.1,1))#, range(40,45+1))
# initial_states = np.meshgrid(np.arange(50,50.1,1), np.arange(10,25,0.1), np.arange(3,3.1,1), np.arange(2,2.1,1))#, range(40,45+1))
# initial_states = np.meshgrid(np.arange(50,50.1,1), np.arange(1.3/np.tan(0.08) - 1, 1.3/np.tan(0.08)+1,0.01), np.arange(4.008555235403628,4.1,1), np.arange(1.3,1.4,1))#, range(40,45+1))
initial_states = np.meshgrid(np.arange(50,50.1,1), np.arange(10, 25, 0.1), np.arange(4.008555235403628,4.1,1), np.arange(1.3,1.4,1))#, range(40,45+1))
#initial_states = np.meshgrid(np.arange(53,53.1,1), np.arange(10,25,0.01), np.arange(53*np.tan(0.08),53*np.tan(0.08)+0.1,1), np.arange(1.9,2.0,1))#, range(40,45+1))
# initial_states = list(np.stack([x.reshape(-1) for x in initial_states[::-1]], axis=1))
initial_states = list(np.stack([x.reshape(-1) for x in initial_states], axis=1))

x = [x[1] for x in initial_states]

# import pdb; pdb.set_trace()
# evaluate_single_state(initial_states[0], 50)
from IPython import embed; embed()
# time_horizons = range(1, 12)
# time_horizons = [11, ]
time_horizons = [10, ]
probs = []
for T in time_horizons:
    with Pool(processes=32) as pool:
        prob = list(tqdm.tqdm(pool.imap(partial(evaluate_single_state, budget=T), initial_states), total=len(initial_states)))
        probs.append(prob)

# x = [x[-1] for x in initial_states]
# x = [x[0] for x in initial_states]

from IPython import embed; embed()
import matplotlib.pyplot as plt
for prob, T in zip(probs, time_horizons):
    plt.plot(x,prob,label='budget=%d'%T)
plt.legend()
plt.show()
# from utils.general_utils import savepklz
# savepklz({'prob':prob, 'initial_states':initial_states}, '/tmp/Slplatoon3_n4_local.pklz')
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np
# from utils.general_utils import savepklz, loadpklz
#
# data = loadpklz('/home/sundw/prob_Sl3_n3_smooth.pklz')
# initial_states = data['initial_states']
# prob = data['prob']
# initial_states = np.array(initial_states)
# dis1 = initial_states[:,0] - initial_states[:,1]
# dis2 = initial_states[:,1] - initial_states[:,2]
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# ax.plot_trisurf(dis1, dis2, prob, linewidth=0.2, antialiased=True)
#
# plt.show()
