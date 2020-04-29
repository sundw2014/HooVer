# HooVer: a statistical model checking tool with optimistic optimization

HooVer uses optimistic optimization to solve statistical model checking problems for MDPs.

### Requirements
HooVer uses Python 3. To install the requirements:
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
The users can create their own model file, put it into the ```models/``` folder, and mofify ```models/__init__.py``` correspondingly. For example, one can create ```models/MyModel.py``` and run HooVer with ```python3 example.py --model MyModel --budget 100000 --nRuns 1```.

In the model file, the user has to create a class which is a subclass of ```NiMC```. Here, we take ```models/MyModel.py``` as an example:
```python
class MyModel(NiMC):
    def __init__(self, sigma, k=10):
        super(MyModel, self).__init__()
```
Then the user has to specify several essential components for this model. First, the user has to set the time bound ```k``` and the set initial states ```Theta``` by calling ```set_k()``` and ```set_Theta``` respectively:
```python
        self.set_Theta([[1,2],[2,3]])
        self.set_k(k)
```

The above code defines an intial state space ![\{(x,y)|x \in \[1,2\], y\in\[2,3\]\}](https://render.githubusercontent.com/render/math?math=%5C%7B(x%2Cy)%7Cx%20%5Cin%20%5B1%2C2%5D%2C%20y%5Cin%5B2%2C3%5D%5C%7D).

Then, the user has to implement the function ```is_usafe()``` which is used to check whether a state is unsafe. This function should return ```True``` if the state is unsafe and return ```False``` otherwise:
```python
    def is_unsafe(self, state):
        if np.linalg.norm(state) > 4:
            return True # return unsafe if norm of the state is greater than 4.
        return False # return safe otherwise.```
```
Finally, the user has to specifing the transition kernel of the model by implementing the function ```transition```. For example, the following code in ```models/MyModel.py``` describes a random motion system:
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

### Acknowledgements

The MFTreeSearchCV code base was developed by Rajat Sen: ( https://github.com/rajatsen91/MFTreeSearchCV ) which in-turn was built on the blackbox optimization code base of Kirthivasan Kandasamy: ( https://github.com/kirthevasank )
