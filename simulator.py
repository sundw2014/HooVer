import sys
sys.path.append('..')
import time
import zmq
import numpy as np
import msgpack

import sys
import argparse
from utils.general_utils import temp_seed

import models

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='')

parser.add_argument('--model', metavar='MODEL',
                        default='Slplatoon3',
                        # choices=model_names,
                        help='models available: ' +
                            ' | '.join(model_names) +
                            ' (default: Slplatoon3)')
parser.add_argument('--args', nargs='+', type=float, help='<Optional>')
parser.add_argument('--initial_states', nargs='+', type=float, help='<Optional>')
parser.add_argument('--seed', type=int, default=1024, help='<Optional>')
parser.add_argument('--port', type=int, help='', required=True)

args = parser.parse_args()

np.random.seed(args.seed)

assert (args.initial_states is None or args.seed is None)


if args.model == 'FakeModel':
    if args.args is None:
        raise ValueError('Please specify the s parameter using --args')
    model = models.__dict__[args.model](s = args.args[0])
else:
    model = models.__dict__[args.model]()

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:"+str(args.port))

def random_initialization(seed, initial_states=None):
    if initial_states is not None:
        state = initial_states
    else:
        with temp_seed(np.abs(seed) % (2**32)):
            state = np.random.rand(model.Theta.shape[0])
              * (model.Theta[:,1] - model.Theta[:,0])
              + model.Theta[:,0]
        state = state.tolist()
    t = 0.
    print('seed = '+str(seed)+', '+'state = '+str(state))
    return state + [t, model.is_unsafe(state)]

while True:
    #  Wait for next request from client
    message = socket.recv()
    _message = msgpack.unpackb(message, use_list=False, raw=False)
    #print('recv: ', _message)
    state = _message[:-1]
    seed = int(_message[-1])
    # from IPython import embed; embed()
    t = state[-2]
    if t < 0: # t == -1 for requesting initialization
        state = random_initialization(seed)
    else:
        t = t+1
        new_state = model.transition(list(state)[0:-2])
        state = new_state + [t, model.is_unsafe(new_state)))]

    #print('send: ', state)
    #  Send reply back to client
    socket.send(msgpack.packb(state, use_bin_type=True))
