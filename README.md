# HooVer: a statistical model checking tool with optimistic optimization

HooVer uses optimistic optimization to solve statistical model checking problems for MDPs.

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
python example.py --model [Slplatoon2/Slplatoon3/Mlplatoon] --budget [time budget for simulator] --numRuns [number of runs] [--useHOO]
```

For example, to check the Slplatoon3 model with MFHOO, run the following command:
```
python example.py --model Slplatoon3 --budget 1.0 --numRuns 10
```
