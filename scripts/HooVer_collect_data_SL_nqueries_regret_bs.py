# Author: Rajat Sen # Modified by Negin
import sys
sys.path.append('..')
import numpy as np
import subprocess
from subprocess import DEVNULL, STDOUT, check_call
import os, signal

from utils.general_utils import loadpklz, savepklz

# -----------------------------------------------------------------------------

useHOO = True
model = 'Slplatoon3'
T = 11
dim = 4
exp_id = int(sys.argv[1])
port_base = 9100
plasmalab_root = '/home/daweis2/plasmalab-1.4.4/'

if __name__ == '__main__':
    budgets = np.logspace(np.log(0.85 * 1e5)/np.log(2), np.log(8e5)/np.log(2), num=6, base=2).astype('int')
    outputs = []
    num_nodes = []
    # bss = [1, 10, 20, 50, 80, 100, 200, 400, 800]
    bss = [10, 20, 50, 80, 100, 200, 400, 800]
    for budget in budgets:
        for bs in bss[::-1]:
            filename = 'data/HooVer_%s_budget%d_nqueries_regret_bs%d_exp%d.pklz'%(model, budget, bs, exp_id)
            os.system('cd ../; python example.py --nRuns 1 --model %s --budget %d --batch_size %d --filename %s'%(model, budget, bs, filename))
            outputs.append(loadpklz('../'+filename))
            num_nodes.append(budget/bs)

    results = []

    optimal_xs = [o['optimal_xs'][0] for o in outputs]
    original_results = [o['optimal_values'][0] for o in outputs]

    port = port_base + exp_id
    tmp_model_name = 'model_%d'%port
    tmp_spec_name = 'spec_%d'%port
    with open(tmp_model_name, 'w') as f:
        f.write('%d %d %d'%(dim, T+1, port))
    with open(tmp_spec_name, 'w') as f:
        f.write('F<=1000 (T<=%d & US>0)'%T)

    for initial_states in optimal_xs:
        initial_states = initial_states.tolist()
        # The os.setsid() is passed in the argument preexec_fn so
        # it's run after the fork() and before  exec() to run the shell.
        simulator = subprocess.Popen('cd ../; python simulator.py --model %s --port %d --initial_states '%(model, port) + ' '.join([str(s) for s in initial_states]), shell=True, preexec_fn=os.setsid, stdout=DEVNULL)
        output = subprocess.check_output(plasmalab_root+'/plasmacli.sh launch -m '+tmp_model_name+':PythonSimulatorBridge -r '+tmp_spec_name+':bltl -a montecarlo -A "Total samples"=30000', universal_newlines=True, shell=True)
        os.killpg(os.getpgid(simulator.pid), signal.SIGTERM)  # Send the signal to all the process groups

        result = float(output.split('\n')[-3].split('|')[4])
        results.append(result)
        print(result)

    savepklz({'results':results, 'num_nodes': num_nodes, 'bss':bss, 'original_results':original_results}, '../data/HooVer_%s_nqueries_regret_bs_exp%d.pklz'%(model, exp_id))
    # from IPython import embed; embed()
    os.system('rm '+tmp_model_name+' '+tmp_spec_name)