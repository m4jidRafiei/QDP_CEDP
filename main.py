from pm_cedp_qdp.qdp import QDP

if __name__ == '__main__':
    log_name = "BPIC13.xes"
    state_window = 200
    state_direction = "backward"  # backward or forward
    export_csv = log_name[:-4] + "_" + str(state_window) + "_" + state_direction + ".csv"
    recursive = True #This will continue quantifying releases until there is no incomplete trace. Otherwise, only one release is quantified.
    only_complete_traces = False #If you want to only consider the complete traces for generating temporal correlations.
    epsilon = 0.01
    qdp = QDP()
    FPL, BPL, TPL = qdp.apply(log_name,epsilon,export_csv,recursive=recursive,
                              only_complete_traces=only_complete_traces, state_window = state_window, state_direction = state_direction)