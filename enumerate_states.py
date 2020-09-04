import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from functools import partial
from multiprocessing import Pool, TimeoutError
import time
import os
import tqdm

import argparse
import numpy as np
import time
import hoover
from utils.general_utils import loadpklz, savepklz
import random

import models

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model', metavar='MODEL',
                        default='Slplatoon',
                        help='models available: ' +
                            ' | '.join(model_names) +
                            ' (default: Slplatoon)')
    parser.add_argument('--args', nargs='+', type=float, help='<Optional> This can be used to pass special arguments to the model.')
    parser.add_argument('--mult', type=int, default=100, help='')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed for reproducibility. (default: 1024)')
    args = parser.parse_args()

    if args.model == 'ConceptualModel':
        if args.args is None:
            raise ValueError('Please specify the s parameter using --args')
        model = models.__dict__[args.model](s = args.args[0])
    else:
        model = models.__dict__[args.model]()

nimc = model

def evaluate_single_state(X, mult=args.mult):
    value = 0
    for i in range(mult):
        reward = nimc(X)
        value = value + reward
    value = value / mult
    return value

initial_states = np.meshgrid(np.arange(20,20.1,1), np.arange(10,10.1,1), np.arange(0,0.1,1), np.arange(0.5,1.5,0.01), np.arange(0.5,1.5,0.01))#, range(40,45+1))
initial_states = list(np.stack([x.reshape(-1) for x in initial_states], axis=1))

# x = [x[1] for x in initial_states]

# import pdb; pdb.set_trace()
# from IPython import embed; embed()
with Pool(processes=32) as pool:
    prob = list(tqdm.tqdm(pool.imap(partial(evaluate_single_state), initial_states), total=len(initial_states)))

from IPython import embed; embed()
# for prob, T in zip(probs, time_horizons):
#     plt.plot(x,prob,label='budget=%d'%T)
# plt.legend()
# plt.show()
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