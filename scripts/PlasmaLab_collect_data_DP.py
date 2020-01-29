import sys
sys.path.append('..')
import numpy as np
import subprocess
from subprocess import DEVNULL, STDOUT, check_call
import os, signal

from utils.general_utils import loadpklz, savepklz

model = 'DetectingPedestrian'
T = 50
dim = 4
exp_id = int(sys.argv[1])
port_base = 9100
plasmalab_root = '/home/daweis2/plasmalab-1.4.4/'

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
    num_queries = budgets
    original_results = []

    for budget in budgets:
        #delta = 2 / np.exp((budget*0.8)*2*(epsilon**2))
        epsilon = np.sqrt(np.log(2/delta)/np.log(np.e)/2/(budget*0.8))
        print(epsilon, delta, budget)
        # The os.setsid() is passed in the argument preexec_fn so
        # it's run after the fork() and before  exec() to run the shell.
        simulator = subprocess.Popen('cd ../; python simulator.py --model %s --port %d'%(model, port), shell=True, preexec_fn=os.setsid, stdout=DEVNULL)
        output = subprocess.check_output(plasmalab_root+'/plasmacli.sh launch -m '+tmp_model_name+':PythonSimulatorBridge -r '+tmp_spec_name+':bltl -a smartsampling -A"Maximum"=True -A"Epsilon"=%lf -A"Delta"=%lf -A"Budget"=%d'%(epsilon, delta, budget), universal_newlines=True, shell=True)
        os.killpg(os.getpgid(simulator.pid), signal.SIGTERM)  # Send the signal to all the process groups
        with open('../data/PlasmaLab_%s_epsilon%lf_delta%lf_budget%d_exp%d.txt'%(model, epsilon, delta, budget, exp_id), 'w') as f:
            f.write(output)

        with open('../data/PlasmaLab_%s_epsilon%lf_delta%lf_budget%d_exp%d.txt'%(model, epsilon, delta, budget, exp_id), 'r') as f:
            output = f.readlines()

        # Strips the newline character
        output = [line.strip() for line in output]
        seeds = output[1:-6]
        final_iter = [int(line.split(' ')[3]) for line in seeds[-budget+10::]]
        final_iter = set(final_iter)
        original_results.append(float(output[-2].split('|')[2]))

        tmp_results = []
        for seed in final_iter:
            # The os.setsid() is passed in the argument preexec_fn so
            # it's run after the fork() and before  exec() to run the shell.
            simulator = subprocess.Popen('cd ../; python simulator.py --model %s --port %d --seed %d'%(model, port, seed), shell=True, preexec_fn=os.setsid, stdout=DEVNULL)
            output = subprocess.check_output(plasmalab_root+'/plasmacli.sh launch -m '+tmp_model_name+':PythonSimulatorBridge -r '+tmp_spec_name+':bltl -a montecarlo -A "Total samples"=300', universal_newlines=True, shell=True)
            os.killpg(os.getpgid(simulator.pid), signal.SIGTERM)  # Send the signal to all the process groups
            tmp_results.append(float(output.split('\n')[-3].split('|')[4]))
        results.append(np.max(tmp_results))

savepklz({'results':results, 'num_queries':num_queries, 'original_results':original_results}, '../data/PlasmaLab_%s_exp%d.pklz'%(model, exp_id))
os.system('rm '+tmp_model_name+' '+tmp_spec_name)
