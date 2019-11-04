# import sys
# sys.path.append('..')
import argparse
import numpy as np
import time
import importlib
import hoover

if __name__ == '__main__':
    model_names = ['Slplatoon2', 'Slplatoon3', 'Mlplatoon']
    true_max_probs = [0.8799573347, 0.8726806826, 0.5918981603]
    true_max_probs = dict(zip(model_names, true_max_probs))

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model', metavar='MODEL',
                        default='Slplatoon3',
                        choices=model_names,
                        help='models available: ' +
                            ' | '.join(model_names) +
                            ' (default: Slplatoon3)')
    parser.add_argument('--useHOO', action='store_true', help='Use HOO')
    parser.add_argument('--numRuns', type=int, help='number of runs')
    parser.add_argument('--budget', type=float, help='time budget for simulator')
    args = parser.parse_args()

    model = importlib.import_module('models.'+args.model)

    num_exp = args.numRuns
    direct_budget = args.budget
    useHOO = args.useHOO

    np.random.seed(1024)

    run_times = []
    memory_usages = []
    num_queries = []
    depths = []
    max_probs = []

    for _ in range(num_exp):
        start_time = time.time()
        MCh_object = model.get_mch_as_mf()
        mf_MCh_object = model.MFMarkovChain.mf_markov_chain(MCh_object)

        noise_var = 5
        sigma = np.sqrt(noise_var)
        nu = 1.0
        rho = 0.95
        C = 0.1
        t0 = mf_MCh_object.opt_fidel_cost

        try:
            Number_of_Queries, max_prob, depth, memory_usage = hoover.estimate_max_probability(mf_MCh_object, nu, rho, sigma, C, t0, direct_budget=direct_budget, useHOO=useHOO)
        except AttributeError as e:
            print(e)
            continue
        end_time = time.time()
        run_time = end_time - start_time

        run_times.append(run_time)
        memory_usages.append(memory_usage/1024.0/1024.0)
        num_queries.append(Number_of_Queries)
        max_probs.append(max_prob)
        depths.append(depth)

    print('budget: ' + str(direct_budget))
    print('running time (s): %.2f +/- %.3f'%(np.mean(run_times), np.std(run_times)))
    print('memory usage (MB): %.2f +/- %.3f'%(np.mean(memory_usages), np.std(memory_usages)))
    print('max probability: %.4f +/- %.5f'%(np.mean(max_probs), np.std(max_probs)))
    print('true max probability: %.5f'%(true_max_probs[args.model]))
    print('number of queries: %d +/- %.1f'%(np.mean(num_queries), np.std(num_queries)))
    print('depth: ' + str(depths))
