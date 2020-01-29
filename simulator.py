import sys
sys.path.append('..')
import time
import zmq
import numpy as np
import msgpack
import importlib

import sys
import argparse

model_names = ['Slplatoon3', 'Mlplatoon', 'DetectingPedestrian', 'Merging', 'FakeModel']

parser = argparse.ArgumentParser(description='')

parser.add_argument('--model', metavar='MODEL',
                        default='Slplatoon3',
                        # choices=model_names,
                        help='models available: ' +
                            ' | '.join(model_names) +
                            ' (default: Slplatoon3)')
parser.add_argument('--initial_states', nargs='+', type=float, help='<Optional>')
parser.add_argument('--seed', type=int, help='<Optional>')
parser.add_argument('--port', type=int, help='', required=True)

args = parser.parse_args()

assert (args.initial_states is None or args.seed is None)

if 'FakeModel' in args.model:
    s = float(args.model[10:])
    simulator = importlib.import_module('models.FakeModel')
    simulator.s = s
else:
    simulator = importlib.import_module('models.'+args.model)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:"+str(args.port))

np.random.seed(1024)

def random_initialization(seed, initial_states=None):
    if initial_states is not None:
        state = initial_states
    else:
        np.random.seed(np.abs(seed) % (2**32))
        state = np.random.rand(len(simulator.state_start)) * simulator.state_range + simulator.state_start
        state = state.tolist()
    t = 1.
    print('seed = '+str(seed)+', '+'state = '+str(state))
    return state + [t, simulator.is_unsafe(state)]

while True:
    #  Wait for next request from client
    message = socket.recv()
    _message = msgpack.unpackb(message, use_list=False, raw=False)
    #print('recv: ', _message)
    state = _message[:-1]
    seed = int(_message[-1])
    # from IPython import embed; embed()
    if int(state[-2]) == 0: # t == 0
        if args.seed is not None:
            state = random_initialization(args.seed)
        elif args.initial_states is not None:
            state = random_initialization(0, args.initial_states)
        else:
            state = random_initialization(seed)
    else:
        state = simulator.step_forward(list(state))

    #print('send: ', state)
    #  Send reply back to client
    socket.send(msgpack.packb(state, use_bin_type=True))
