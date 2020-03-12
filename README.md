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
The users can create their own model following the interface used in ```models/Slplatoon3.py```. Data and functions to be implemented include ```T```, ```state_start```, ```state_range```, ```is_unsafe```, and ```step_forward```.

```T``` is the time horizon of the model.

```state_start``` and ```state_range``` describe a hyper-rectangle which is the initial state domain.

```is_usafe``` is used to determine whether a state is unsafe.

```step_forward``` is a single-step transition function of the model.

### Acknowledgements

The MFTreeSearchCV code base was developed by Rajat Sen: ( https://github.com/rajatsen91/MFTreeSearchCV ) which in-turn was built on the blackbox optimization code base of Kirthivasan Kandasamy: ( https://github.com/kirthevasank )
