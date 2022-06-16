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
    log_name = "Sepsis Cases.xes"
    max_num_release = 5
    state_window = 200              # A large number will consider the entire prefix/suffix of traces
    state_direction = "backward"    # Backward (prefix) or forward (suffix)
    event_percentage = 0.5          # The percentage of events included in the first release
    recursive = True                # This will continue quantifying releases until there is no incomplete trace or reaching max_num_release. Otherwise, only one release is quantified.
    only_complete_traces = False    # If one wants to only consider the complete traces for generating temporal correlations.
    epsilon = 0.01                  # The privacy parameter of DP mechanism
    window_size = 2                 # How many new events per trace are released at each publish.
    certain_release = True          # If this is true, the number of new events per variant in each new release is equal to window_size, otherwise it is a random number range from 0 to window_size
    explore_depth = window_size     # The depth of exploring the transitions system.
    export_csv = log_name[:-4] + "_" + str(state_window) + "_" + state_direction + "_" + str(certain_release) + str(window_size) + ".csv"

    qdp = QDP()
    release_index, BPL_list, FPL_list, TPL_list = qdp.apply(log_name, epsilon, export_csv, recursive=recursive,
                              only_complete_traces=only_complete_traces, state_window=state_window,
                              state_direction=state_direction,
                              explore_depth = explore_depth,
                              window_size= window_size,
                              certain_release= certain_release,
                              event_percentage= event_percentage,
                              max_num_release = max_num_release)
```
