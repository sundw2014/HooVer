# HooVer: a statistical model checking tool with optimistic optimization

HooVer uses optimistic optimization to solve statistical model checking problems for MDPs.

### Requirements
HooVer uses Python3. To install the requirements:
```
pip3 install -r requirements.txt
```

### Usage
```
python3 example.py --model [Slplatoon3/Mlplatoon/Merging/DetectingPedestrian] --budget [budget for simulator] --nRuns [number of runs] --rho_max [the rho_max parameter] --sigma [the sigma parameter for UCB] --nHOOs [number of HOO instances to use] --batch_size [batch size] --filename [path to save the results]
```

For example, to check the Mlplatoon model, run the following command:
```
python3 example.py --model Mlplatoon --budget 8000000 --nRuns 1 --sigma 1e-5
```

### Verify your own model
The users can create their own model and put it into the ```models/``` folder. For example, one can create ```models/MyModel.py``` and run HooVer with ```python3 example.py --model MyModel --budget 100000 --nRuns 1```. The new model should implement the interface used in ```models/Slplatoon3.py```. Specifically, variables and functions to be implemented include ```T```, ```state_start```, ```state_range```, ```is_unsafe```, and ```step_forward```.

```T``` is the time horizon of the model. For example, ```T = 10```.

```state_start``` and ```state_range``` describe a hyper-rectangle which is the initial state domain. For example,
```python
state_start = np.array([1,2])
state_range = np.array([1,1])
```
the above code defines an intial state space ![\{(x,y)|x \in \[1,2\], y\in\[2,3\]\}](https://render.githubusercontent.com/render/math?math=%5C%7B(x%2Cy)%7Cx%20%5Cin%20%5B1%2C2%5D%2C%20y%5Cin%5B2%2C3%5D%5C%7D).

```is_usafe``` is used to check whether a state is unsafe. This function should return ```1.``` if the state is unsafe and return ```0.``` otherwise. For example,
```python
def is_unsafe(state):
    if np.linalg.norm(state) > 4:
        return 1. # return unsafe if norm of the state is greater than 4.
    return 0. # return safe otherwise.
```

```step_forward``` is the single-step transition function of the model. For example, the following code models a Brownian motion process
```python
def step_forward(state):
    # The input state variable contains the state of the system, the current time step, and the isunafe flag, i.e. state = system_state + [t, is_unsafe(system_state)]
    system_state = state[:-2] # extract the state of the system
    t = state[-2] # extract the current time step
    system_state = np.array(system_state)
    system_state += 1.0 * np.random.randn(len(system_state)) # a normally distributed increment
    system_state = system_state.tolist()
    t += 1 # increase the time step by 1
    return system_state + [t, is_unsafe(system_state)] # return the new state
```

### Acknowledgements

The MFTreeSearchCV code base was developed by Rajat Sen: ( https://github.com/rajatsen91/MFTreeSearchCV ) which in-turn was built on the blackbox optimization code base of Kirthivasan Kandasamy: ( https://github.com/kirthevasank )
