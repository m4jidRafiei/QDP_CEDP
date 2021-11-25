## Introduction
This project implements the quantification of privacy leakage for differential privacy mechanisms in continuous event data publishing.
## Python package
The implementation has been published as a standard Python package. Use the following command to install the corresponding Python package:

```shell
pip install pm-cedp-qdp
```

## Usage
```python
from pm_cedp_qdp.qdp import QDP

if __name__ == '__main__':
    log_name = "BPI2012App.xes"
    state_window = 200 # a large number will consider the entire prefix/suffix of traces
    state_direction = "backward"  # backward (prefix) or forward (suffix)
    export_csv = log_name[:-4] + "_" + str(state_window) + "_" + state_direction + ".csv"
    recursive = True #This will continue quantifying releases until there is no incomplete trace. Otherwise, only one release is quantified.
    only_complete_traces = False #If you want to only consider the complete traces for generating temporal correlations.
    epsilon = 0.01
    qdp = QDP()
    FPL, BPL, TPL = qdp.apply(log_name,epsilon,export_csv,recursive=recursive,
                              only_complete_traces=only_complete_traces, state_window = state_window, state_direction = state_direction)
```
