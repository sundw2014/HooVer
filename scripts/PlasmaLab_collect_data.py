import subprocess
from subprocess import DEVNULL, STDOUT, check_call
import os, signal

import sys
sys.path.append('..')
from utils.general_utils import loadpklz, savepklz

import numpy as np

plasmalab_root = '/home/daweis2/plasmalab-1.4.4/'
# FIXME
exp_name = 'Mlplatoon_n3l3'
#exp_name = 'AvoidingPedestrian'
#exp_name = 'FakeModel'

#budgets = (np.array([2000000, 1000000, 800000, 600000, 400000, 200000, 100000, 80000, 50000, 30000, 10000]) / 16).astype('int').tolist()
budgets = (np.logspace(np.log(8000)/np.log(2), np.log(800000)/np.log(2), num=10, base=2) / 16).astype('int').tolist()[::-1]
#budgets = (np.array([200000, 100000, 80000, 50000, 30000, 10000]) / 16).astype('int').tolist()
#epsilon = 0.01
delta = 0.01

tmp_model_name = 'model_'+sys.argv[1]
# FIXME
with open(tmp_model_name, 'w') as f:
    f.write('18 11 '+sys.argv[1])

results = []
num_queries = []
original_results = []

for budget in budgets[::-1]:
    #delta = 2 / np.exp((budget*0.8)*2*(epsilon**2))
    epsilon = np.sqrt(np.log(2/delta)/np.log(np.e)/2/(budget*0.8))
    print(epsilon, delta, budget)
    # The os.setsid() is passed in the argument preexec_fn so
    # it's run after the fork() and before  exec() to run the shell.
    simulator = subprocess.Popen('python simulator.py --port '+sys.argv[1], shell=True, preexec_fn=os.setsid, stdout=DEVNULL)
    output = subprocess.check_output(plasmalab_root+'/plasmacli.sh launch -m '+tmp_model_name+':PythonSimulatorBridge -r spec:bltl -a smartsampling -A"Maximum"=True -A"Epsilon"=%lf -A"Delta"=%lf -A"Budget"=%d'%(epsilon, delta, budget), universal_newlines=True, shell=True)
    os.killpg(os.getpgid(simulator.pid), signal.SIGTERM)  # Send the signal to all the process groups
    with open('plasmalab_output/'+exp_name+'_port%s_epsilon%lf_delta%lf_budget%d.txt'%(sys.argv[1], epsilon, delta, budget), 'w') as f:
        f.write(output)

    with open('plasmalab_output/'+exp_name+'_port%s_epsilon%lf_delta%lf_budget%d.txt'%(sys.argv[1], epsilon, delta, budget), 'r') as f:
        output = f.readlines()

    # Strips the newline character
    output = [line.strip() for line in output]
    seeds = output[1:-6]
    final_iter = [int(line.split(' ')[3]) for line in seeds[-budget+10::]]
    final_iter = set(final_iter)
    num_queries.append(len(seeds))
    original_results.append(float(output[-2].split('|')[2]))

    tmp_results = []
    for seed in final_iter:
        # The os.setsid() is passed in the argument preexec_fn so
        # it's run after the fork() and before  exec() to run the shell.
        simulator = subprocess.Popen('python simulator.py --port %s --seed %d'%(sys.argv[1], seed), shell=True, preexec_fn=os.setsid, stdout=DEVNULL)
        output = subprocess.check_output(plasmalab_root+'/plasmacli.sh launch -m '+tmp_model_name+':PythonSimulatorBridge -r spec:bltl -a montecarlo -A "Total samples"=30000', universal_newlines=True, shell=True)
        os.killpg(os.getpgid(simulator.pid), signal.SIGTERM)  # Send the signal to all the process groups
        tmp_results.append(float(output.split('\n')[-3].split('|')[4]))
    results.append(np.max(tmp_results))

savepklz({'results':results, 'num_queries':num_queries, 'original_results':original_results}, 'plasmalab_output/'+exp_name+'_port%s_results_vs_num_queries.pklz'%(sys.argv[1]))
from IPython import embed; embed()
os.system('rm '+tmp_model_name)
