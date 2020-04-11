# import sys
# sys.path.append('..')
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
    # true_max_probs = dict(zip(model_names, true_max_probs))

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model', metavar='MODEL',
                        default='Slplatoon3',
                        # choices=model_names,
                        help='models available: ' +
                            ' | '.join(model_names) +
                            ' (default: Slplatoon3)')
    parser.add_argument('--args', nargs='+', type=float, help='<Optional>')
    parser.add_argument('--nRuns', type=int, default=1, help='number of runs')
    parser.add_argument('--budget', type=int, default=int(1e6), help='budget for number of simulations')
    parser.add_argument('--rho_max', type=float, default=0.6, help='time budget for simulator')
    parser.add_argument('--sigma', type=float, help='Sigma parameter for UCB')
    parser.add_argument('--nHOOs', type=int, default=4, help='number of HOO instances to use')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--filename', type=str, default='./output.pklz', help='path to save the results')
    parser.add_argument('--seed', type=int, default=1024, help='random seed for reproducibility')
    args = parser.parse_args()

    if args.model == 'FakeModel':
        if args.args is None:
            raise ValueError('Please specify the s parameter using --args')
        model = models.__dict__[args.model](s = args.args[0])
    else:
        model = models.__dict__[args.model]()

    num_exp = args.nRuns

    budget_for_each_HOO = (args.budget - 40000) / args.nHOOs / args.batch_size

    running_times = []
    memory_usages = []
    depths = []
    optimal_xs = []
    optimal_values = []
    num_nodes = []

    random.seed(args.seed)
    np.random.seed(args.seed)
    for _ in range(num_exp):
        start_time = time.time()
        # import pdb; pdb.set_trace()
        nimc = model

        if args.sigma is None:
            sigma = np.sqrt(0.5*0.5/args.batch_size)
        else:
            sigma = args.sigma

        rho_max = args.rho_max

        try:
            optimal_x, optimal_value, depth, memory_usage, n_nodes =\
             hoover.estimate_max_probability(nimc, args.nHOOs, rho_max, sigma, budget_for_each_HOO, args.batch_size)
        except AttributeError as e:
            print(e)
            continue

        end_time = time.time()
        running_time = end_time - start_time

        running_times.append(running_time)
        memory_usages.append(memory_usage/1024.0/1024.0)
        optimal_values.append(optimal_value)
        optimal_xs.append(optimal_x)
        depths.append(depth)
        num_nodes.append(n_nodes)

    print('budget: ' + str(args.budget))
    print('running time (s): %.2f +/- %.3f'%(np.mean(running_times), np.std(running_times)))
    print('memory usage (MB): %.2f +/- %.3f'%(np.mean(memory_usages), np.std(memory_usages)))
    print('optimal_values: %.4f +/- %.5f'%(np.mean(optimal_values), np.std(optimal_values)))
    print('optimal_xs: '+str(optimal_xs))
    print('depth: ' + str(depths))
    print('number of nodes: ' + str(num_nodes))
    savepklz({'budget':args.budget, 'running_times':running_times, 'memory_usages':memory_usages, 'optimal_values':optimal_values, 'optimal_xs':optimal_xs, 'depths':depths, 'num_nodes':num_nodes}, args.filename)
