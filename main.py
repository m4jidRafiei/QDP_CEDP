from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.transition_system import algorithm as ts_discovery
from pm_cedp_qdp.utilities import utilities
from pm_cedp_qdp.qdp import QDP
import pm4py
import os

if __name__ == '__main__':
    epsilon = 0.01
    BPL = 0.01
    FPL = 0.01
    TPL = BPL + FPL - epsilon
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(current_dir,'event_logs',"running_example.xes")
    log = xes_importer.apply(log_path)

    util = utilities()
    qdp = QDP()

    #-------start_point is the point in time when all the cases started, one can force having complete traces included.-------
    start_point = util.get_start_point(log)
    # start_point = "2011-01-15 16:04:43" #for running_example
    # start_point = "2012-05-07 16:04:43" #for BPIC2013

    #--------adding artificial start ("▶") and end ("■") activities-------
    log = pm4py.insert_artificial_start_end(log)

    #--------initial_log is the first published event log where all the traces started--------
    #--------cont_log is the next pulished event log where one event is published per each previously incomplete published trace----
    initial_log, cont_log = qdp.split_log_into_initial_and_continuous(log,start_point)

    #-------If you want to only consider the complete traces for generating temporal correlations------
    initial_log_complete = util.remove_incomplete_traces(initial_log)

    FPL_valid = False
    if len(initial_log_complete) >= len(initial_log) /2:
        FPL_valid = True
    #------Generating transition system for calculating temporal correlations-------
    ts = ts_discovery.apply(initial_log_complete, parameters={'direction': "backward", 'view': "sequence", 'window': 200, 'include_data': True})

    #------Calculating backward anf forward privacy leakages based on transition system ----------
    forward_dict, backward_dict, state_with_id = qdp.probability_matrices_non_sparse(ts)

    #------Keeping the probability information of the states that we have in the cont_log---------
    filtered_forward = qdp.probability_dict_filter(forward_dict, cont_log, state_with_id)
    filtered_backward = qdp.probability_dict_filter(backward_dict, cont_log, state_with_id)

    #-----Calculating comulative DP disclosure because of temporal correlations-------
    if FPL_valid:
        FPL = qdp.comulative_leakage_calc_dict(filtered_forward,epsilon,FPL)
    else:
        FPL = epsilon
    BPL = qdp.comulative_leakage_calc_dict(filtered_backward,epsilon,BPL)

    TPL = FPL + BPL - epsilon

    print("FPL:" + str(FPL) + " - BPL:" + str(BPL) + " - TPL:" + str(TPL))

    FPL, BPL, TPL = qdp.recursive_comulative_leakage_calc_dict(log,cont_log,epsilon,FPL,BPL)


    # -----This implementation is super slow because of sparse matrices-----------
    # df_forward,df_backward,state_with_id = qdp.probability_matrices(ts)
    # filtered_forward = qdp.probability_matrices_filter(df_forward, cont_log, state_with_id)
    # filtered_backward = qdp.probability_matrices_filter(df_backward, cont_log, state_with_id)
    # FPL = qdp.commulative_leakage_calc_matrix(ts,filtered_forward,0.01,0.01)
    # BPL = qdp.commulative_leakage_calc_matrix(ts,filtered_backward,0.01,0.01)

    #-----visualization-------
    # viz = ts_vis.apply(ts, parameters={ts_vis.Variants.VIEW_BASED.value.Parameters.FORMAT: "svg"})
    # ts_vis.view(viz)