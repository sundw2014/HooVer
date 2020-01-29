# HooVer: a statistical model checking tool with optimistic optimization

HooVer uses optimistic optimization to solve statistical model checking problems for MDPs. Please refer to this [technical report](http://mitras.ece.illinois.edu/research/2019/oosmc-full.pdf) for more details.

```
@TECHREPORT{SMCOO:2020,
  author = {Negin musavi and Dawei Sun and Sayan Mitra and Geir Dullerud and Sanjay Shakkottai},
  title = {Statistical Model Checking with Optimistic Optimization},
  institution = {University of Illinois at Urbana Champaign},
  year = {2019},
  month = {October},
  note = {Available from \url{http://mitras.ece.illinois.edu/research/2019/oosmc-full.pdf}}
}
```

### Requirements
Install the requirements:
```
pip install -r requirements.txt
```

### Usage
```
python example.py --model [Slplatoon3/Mlplatoon/Merging/DetectingPedestrian] --budget [budget for simulator] --nRuns [number of runs] --rho_max [the rho_max parameter] --sigma [the sigma parameter for UCB] --nHOOs [number of HOO instances to use] --batch_size [batch size] --filename [path to save the results]
```

For example, to check the Mlplatoon model, run the following command:
```
python example.py --model Mlplatoon --budget 8000000 --numRuns 1 --sigma 1e-5
```

### Verify your own model
The users can create their own model following the interface used in ```models/Slplatoon3.py```. Data and functions to be implemented include ```T```, ```state_start```, ```state_range```, ```is_unsafe```, and ```step_forward```.

```T``` is the time horizon of the model.

```state_start``` and ```state_range``` describe a hyper-rectangle which is the initial state domain.

```is_usafe``` is used to determine whether a state is unsafe.

```step_forward``` is a single-step transition function of the model.

### Acknowledgements

The MFTreeSearchCV code base was developed by Rajat Sen: ( https://github.com/rajatsen91/MFTreeSearchCV ) which in-turn was built on the blackbox optimization code base of Kirthivasan Kandasamy: ( https://github.com/kirthevasank )
