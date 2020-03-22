# import sys
# sys.path.append('..')
import argparse
import numpy as np
import time
import importlib
import hoover
import MFMC
from utils.general_utils import loadpklz, savepklz
import random

if __name__ == '__main__':
    model_names = ['Slplatoon3', 'Mlplatoon', 'DetectingPedestrian', 'Merging', 'FakeModel']
    # true_max_probs = dict(zip(model_names, true_max_probs))

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model', metavar='MODEL',
                        default='Slplatoon3',
                        # choices=model_names,
                        help='models available: ' +
                            ' | '.join(model_names) +
                            ' (default: Slplatoon3)')
    parser.add_argument('--nRuns', type=int, default=1, help='number of runs')
    parser.add_argument('--budget', type=int, default=int(1e6), help='budget for number of simulations')
    parser.add_argument('--rho_max', type=float, default=0.6, help='time budget for simulator')
    parser.add_argument('--sigma', type=float, help='Sigma parameter for UCB')
    parser.add_argument('--nHOOs', type=int, default=4, help='number of HOO instances to use')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--filename', type=str, default='./output.pklz', help='path to save the results')
    parser.add_argument('--seed', type=int, default=1024, help='random seed for reproducibility')
    args = parser.parse_args()

    if 'FakeModel' in args.model:
        s = float(args.model[10:])
        simulator = importlib.import_module('models.FakeModel')
        simulator.s = s
        print(simulator.s)
    else:
        simulator = importlib.import_module('models.'+args.model)

    MFMC.set_simulator(simulator)

    num_exp = args.nRuns

    budget_for_each_HOO = (args.budget - 40000) / args.nHOOs / args.batch_size

    running_times = []
    memory_usages = []
    depths = []
    optimal_xs = []
    optimal_values = []

    random.seed(args.seed)
    np.random.seed(args.seed)
    for _ in range(num_exp):
        start_time = time.time()
        # import pdb; pdb.set_trace()
        MCh_object, MCh = MFMC.get_mch_as_mf(args.batch_size)
        mf_MCh_object = MFMC.MFMarkovChain.mf_markov_chain(MCh_object)

        if args.sigma is None:
            sigma = np.sqrt(0.5*0.5/args.batch_size)
        else:
            sigma = args.sigma

        rho_max = args.rho_max

        try:
            optimal_x, optimal_value, depth, memory_usage = hoover.estimate_max_probability(mf_MCh_object, args.nHOOs, rho_max, sigma, budget_for_each_HOO)
        except AttributeError as e:
            print(e)
            continue

        end_time = time.time()
        running_time = end_time - start_time

        running_times.append(running_time)
        memory_usages.append(memory_usage/1024.0/1024.0)
        optimal_values.append(optimal_value)
        optimal_xs.append(MFMC.get_full_state(optimal_x))
        depths.append(depth)

    print('budget: ' + str(args.budget))
    print('running time (s): %.2f +/- %.3f'%(np.mean(running_times), np.std(running_times)))
    print('memory usage (MB): %.2f +/- %.3f'%(np.mean(memory_usages), np.std(memory_usages)))
    print('optimal_values: %.4f +/- %.5f'%(np.mean(optimal_values), np.std(optimal_values)))
    print('optimal_xs: '+str(optimal_xs))
    print('depth: ' + str(depths))
    savepklz({'budget':args.budget, 'running_times':running_times, 'memory_usages':memory_usages, 'optimal_values':optimal_values, 'optimal_xs':optimal_xs, 'depths':depths}, args.filename)
