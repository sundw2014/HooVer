import sys
sys.path.append('..')
import numpy as np
import subprocess
from subprocess import DEVNULL, STDOUT, check_call
import os, signal

from utils.general_utils import loadpklz, savepklz, evaluate_single_state, temp_seed

import models

model = 'DetectingPedestrian'
nimc = models.__dict__[model]()

T = nimc.k
dim = nimc.Theta.shape[0]
exp_id = int(sys.argv[1])
port_base = 9100
plasmalab_root = '/home/daweis2/plasmalab-1.4.4/'

def get_initial_state(seed):
    with temp_seed(np.abs(seed) % (2**32)):
        state = np.random.rand(nimc.Theta.shape[0])\
          * (nimc.Theta[:,1] - nimc.Theta[:,0])\
          + nimc.Theta[:,0]
    state = state.tolist()
    return state

if __name__ == '__main__':
    budgets = np.logspace(np.log(0.65 * 1e5)/np.log(2), np.log(8e5)/np.log(2), num=7, base=2).astype('int')
    budgets = (budgets/16.0).astype('int')
    #epsilon = 0.01
    delta = 0.01

    port = port_base + exp_id
    tmp_model_name = 'model_%d'%port
    tmp_spec_name = 'spec_%d'%port
    with open(tmp_model_name, 'w') as f:
        f.write('%d %d %d'%(dim, T+1, port))
    with open(tmp_spec_name, 'w') as f:
        f.write('F<=1000 (T<=%d & US>0)'%T)

    results = []
    original_results = []
    num_queries = []

    for budget in budgets:
        #delta = 2 / np.exp((budget*0.8)*2*(epsilon**2))
        epsilon = np.sqrt(np.log(2/delta)/np.log(np.e)/2/(budget*0.8))
        print(epsilon, delta, budget)
        # The os.setsid() is passed in the argument preexec_fn so
        # it's run after the fork() and before  exec() to run the shell.
        _simulator = subprocess.Popen('cd ../; python3 simulator.py --model %s --port %d'%(model, port), shell=True, preexec_fn=os.setsid, stdout=DEVNULL)
        output = subprocess.check_output(plasmalab_root+'/plasmacli.sh launch -m '+tmp_model_name+':PythonSimulatorBridge -r '+tmp_spec_name+':bltl -a smartsampling -A"Maximum"=True -A"Epsilon"=%lf -A"Delta"=%lf -A"Budget"=%d'%(epsilon, delta, budget), universal_newlines=True, shell=True)
        os.killpg(os.getpgid(_simulator.pid), signal.SIGTERM)  # Send the signal to all the process groups
        with open('../data/PlasmaLab_%s_epsilon%lf_delta%lf_budget%d_exp%d.txt'%(model, epsilon, delta, budget, exp_id), 'w') as f:
            f.write(output)

        with open('../data/PlasmaLab_%s_epsilon%lf_delta%lf_budget%d_exp%d.txt'%(model, epsilon, delta, budget, exp_id), 'r') as f:
            output = f.readlines()

        # Strips the newline character
        output = [line.strip() for line in output]
        seeds = output[1:-6]
        num_queries.append(len(seeds))
        final_iter = [int(line.split(' ')[3]) for line in seeds[-budget+10::]]
        final_iter = set(final_iter)
        original_results.append(float(output[-2].split('|')[2]))
        # print(original_results)
        final_iter = [get_initial_state(seed) for seed in final_iter]
        tmp_results = []
        for initial_states in final_iter:
            np.random.seed(1024)
            result = evaluate_single_state(nimc, initial_states, nimc.k, mult=10000)
            tmp_results.append(result)
        initial_states = final_iter[np.argmax(tmp_results)]
        np.random.seed(1024)
        result = evaluate_single_state(nimc, initial_states, nimc.k, mult=250000)
        print(result)

        results.append(result)
print({'results':results, 'num_queries':num_queries, 'original_results':original_results})
savepklz({'results':results, 'num_queries':num_queries, 'original_results':original_results}, '../data/PlasmaLab_%s_exp%d.pklz'%(model, exp_id))
os.system('rm '+tmp_model_name+' '+tmp_spec_name)
