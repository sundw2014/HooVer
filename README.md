# HooVer: a statistical model checking tool with optimistic optimization

__NOTE: For reviewers who want to reproduce the results in the paper, please see [Reproduce.md](Reproduce.md).__

HooVer uses optimistic optimization to solve statistical model checking problems for MDPs.

### Requirements
HooVer uses Python 3. To install the requirements:
```
pip3 install -r requirements.txt
```

### Usage
```
usage: check.py [-h] [--model MODEL] [--args ARGS [ARGS ...]]
                  [--nRuns NRUNS] [--budget BUDGET] [--rho_max RHO_MAX]
                  [--sigma SIGMA] [--nHOOs NHOOS] [--batch_size BATCH_SIZE]
                  [--output OUTPUT] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         models available: ConceptualModel |
                        DetectingPedestrian | Merging | Mlplatoon | MyModel |
                        Slplatoon (default: Slplatoon)
  --args ARGS [ARGS ...]
                        <Optional> This can be used to pass special arguments
                        to the model.
  --nRuns NRUNS         Number of repetitions. (default: 1)
  --budget BUDGET       Budget for total number of simulations. (default: 1e6)
  --rho_max RHO_MAX     Smoothness parameter. (default: 0.6)
  --sigma SIGMA         <Optional> Sigma parameter for UCB. If not specified,
                        it will be sqrt(0.5*0.5/batch_size).
  --nHOOs NHOOS         Number of HOO instances to use. (default: 4)
  --batch_size BATCH_SIZE
                        Batch size. (default: 100)
  --output OUTPUT       Path to save the results. (default: ./output.pklz)
  --seed SEED           Random seed for reproducibility. (default: 1024)
```

For example, to check the toy model, run the following command:
```
python3 check.py --model MyModel --budget 100000
```
You will find the following in the output, which is the most unsafe initial state:
```
...
optimal_xs: [array([1.97460938, 2.99414062])]
...
```

### Verify your own model
The users can create their own model file, put it into the ```models/``` folder, and mofify ```models/__init__.py``` correspondingly. For example, one can create ```models/MyModel.py``` and run HooVer with ```python3 check.py --model MyModel --budget 100000```.

In the model file, the user has to create a class which is a subclass of ```NiMC```. Here, we take ```models/MyModel.py``` as an example:
```python
class MyModel(NiMC):
    def __init__(self, sigma, k=10):
        super(MyModel, self).__init__()
```
Then the user has to specify several essential components for this model. First, the user has to set the time bound ```k``` and the initial states set ```Theta``` by calling ```set_k()``` and ```set_Theta()``` respectively:
```python
        self.set_Theta([[1,2],[2,3]])
        self.set_k(k)
```

The above code defines an intial state space ![\Theta = \{ (x,y) | x \in \[1,2\], y \in \[2,3\] \}](https://render.githubusercontent.com/render/math?math=%5CTheta%20%3D%20%5C%7B%20(x%2Cy)%20%7C%20x%20%5Cin%20%5B1%2C2%5D%2C%20y%20%5Cin%20%5B2%2C3%5D%20%5C%7D).

Then, the user has to implement the function ```is_usafe()``` which is used to check whether a state is unsafe. This function should return ```True``` if the state is unsafe and return ```False``` otherwise:
```python
    def is_unsafe(self, state):
        if np.linalg.norm(state) > 4:
            return True # return unsafe if norm of the state is greater than 4.
        return False # return safe otherwise.```
```
Finally, the user has to specifing the transition kernel of the model by implementing the function ```transition()```. For example, the following code in ```models/MyModel.py``` describes a random motion system:
```python
    def transition(self, state):
        state = np.array(state)
        # increment is a 2-dimensional
        # normally distributed vector
        increment = self.sigma * np.random.randn(2)
        state += increment
        state = state.tolist()
        return state # return the new state
```

The user also has to update ```models/__init__.py``` by adding a line to import the new model file. For example, ```from .MyModel import *```.

### Acknowledgements

The MFTreeSearchCV code base was developed by Rajat Sen: ( https://github.com/rajatsen91/MFTreeSearchCV ) which in-turn was built on the blackbox optimization code base of Kirthivasan Kandasamy: ( https://github.com/kirthevasank )
