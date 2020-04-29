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
    parser.add_argument('--nRuns', type=int, default=1, help='Number of repetitions. (default: 1)')
    parser.add_argument('--budget', type=int, default=int(1e6), help='Budget for total number of simulations. (default: 1e6)')
    parser.add_argument('--rho_max', type=float, default=0.6, help='Smoothness parameter. (default: 0.6)')
    parser.add_argument('--sigma', type=float, help='<Optional> Sigma parameter for UCB. If not specified, it will be sqrt(0.5*0.5/batch_size).')
    parser.add_argument('--nHOOs', type=int, default=4, help='Number of HOO instances to use. (default: 4)')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size. (default: 100)')
    parser.add_argument('--output', type=str, default='./output.pklz', help='Path to save the results. (default: ./output.pklz)')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed for reproducibility. (default: 1024)')
    args = parser.parse_args()

    if args.model == 'ConceptualModel':
        if args.args is None:
            raise ValueError('Please specify the s parameter using --args')
        model = models.__dict__[args.model](s = args.args[0])
    else:
        model = models.__dict__[args.model]()

    num_exp = args.nRuns

    # calculate budget for each HOO instance
    # we use 10000 simulations for each instance of HOO to do MC estimation
    budget_for_each_HOO = (args.budget - 10000 * args.nHOOs) / args.nHOOs / args.batch_size

    running_times = []
    memory_usages = []
    depths = []
    optimal_xs = []
    optimal_values = []
    num_nodes = []
    n_queries = 0

    # set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    for _ in range(num_exp):
        start_time = time.time()
        nimc = model

        # Calculate parameter sigma for UCB.
        # sqrt(0.5*0.5/args.batch_size) is a valid parameter for any model.
        # The user can also pass smaller sigma parameters to encourage the
        # algorithm to explore deeper in the tree.
        if args.sigma is None:
            sigma = np.sqrt(0.5*0.5/args.batch_size)
        else:
            sigma = args.sigma

        rho_max = args.rho_max

        try:
            # call hoover.estimate_max_probability with the model and parameters
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
        # Get the real number of queries from the model object.
        # It may be a little bit different from the budget.
        n_queries = nimc.cnt_queries
    print('budget: ' + str(args.budget))
    print('running time (s): %.2f +/- %.3f'%(np.mean(running_times), np.std(running_times)))
    print('memory usage (MB): %.2f +/- %.3f'%(np.mean(memory_usages), np.std(memory_usages)))
    print('optimal_values: %.4f +/- %.5f'%(np.mean(optimal_values), np.std(optimal_values)))
    print('optimal_xs: '+str(optimal_xs))
    print('depth: ' + str(depths))
    print('number of nodes: ' + str(num_nodes))
    print('actual n_queries: '+str(n_queries))
    savepklz({'budget':n_queries, 'running_times':running_times, 'memory_usages':memory_usages, 'optimal_values':optimal_values, 'optimal_xs':optimal_xs, 'depths':depths, 'num_nodes':num_nodes}, args.output)
