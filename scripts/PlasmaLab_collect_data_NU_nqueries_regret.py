import sys
sys.path.append('..')
import numpy as np
import subprocess
from subprocess import DEVNULL, STDOUT, check_call
import os, signal

from utils.general_utils import loadpklz, savepklz

model = 'FakeModel'
T = 1
dim = 2
exp_id = int(sys.argv[1])
port_base = 9100
plasmalab_root = '/home/daweis2/plasmalab-1.4.4/'

import models.FakeModel as FakeModel

if __name__ == '__main__':
    ss = np.array([1e-1, 1e-2, 1e-3, 3e-4])
    _budgets = [[0.5, 1.0], [0.5, 1.0], [0.5, 1.5], [1.0, 3.0]]
    _budgets = [(np.logspace(np.log(start * 1e5)/np.log(2), np.log(end * 1e5)/np.log(2), num=10, base=2)/16).astype('int') for start,end in _budgets]

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

    for s, budgets in zip(ss, _budgets):
        s_results = []
        for budget in budgets:
            FakeModel.s = s
            #delta = 2 / np.exp((budget*0.8)*2*(epsilon**2))
            epsilon = np.sqrt(np.log(2/delta)/np.log(np.e)/2/(budget*0.8))
            print(epsilon, delta, budget, s)
            # The os.setsid() is passed in the argument preexec_fn so
            # it's run after the fork() and before  exec() to run the shell.
            simulator = subprocess.Popen('cd ../; python simulator.py --model %s_%lf --port %d'%(model, s, port), shell=True, preexec_fn=os.setsid, stdout=DEVNULL)
            output = subprocess.check_output(plasmalab_root+'/plasmacli.sh launch -m '+tmp_model_name+':PythonSimulatorBridge -r '+tmp_spec_name+':bltl -a smartsampling -A"Maximum"=True -A"Epsilon"=%lf -A"Delta"=%lf -A"Budget"=%d'%(epsilon, delta, budget), universal_newlines=True, shell=True)
            os.killpg(os.getpgid(simulator.pid), signal.SIGTERM)  # Send the signal to all the process groups
            with open('../data/PlasmaLab_%s_nqueries_regret_epsilon%lf_delta%lf_budget%d_s%lf_exp%d.txt'%(model, epsilon, delta, budget, s, exp_id), 'w') as f:
                f.write(output)

            with open('../data/PlasmaLab_%s_nqueries_regret_epsilon%lf_delta%lf_budget%d_s%lf_exp%d.txt'%(model, epsilon, delta, budget, s, exp_id), 'r') as f:
                output = f.readlines()

            # Strips the newline character
            output = [line.strip() for line in output]
            seeds = output[1:-6]
            final_iter = [int(line.split(' ')[3]) for line in seeds[-budget+10::]]
            final_iter = set(final_iter)
            original_results.append(float(output[-2].split('|')[2]))

            tmp_results = []
            print(final_iter)
            for seed in final_iter:
                state = FakeModel.random_initialization(seed)[:-2]
                tmp_results.append(FakeModel.get_prob(state))
            s_results.append(np.max(tmp_results))
        results.append(s_results)
savepklz({'results':results, 'ss':ss, 'budgets':_budgets}, '../data/PlasmaLab_%s_nqueries_regret_exp%d.pklz'%(model, exp_id))
os.system('rm '+tmp_model_name+' '+tmp_spec_name)