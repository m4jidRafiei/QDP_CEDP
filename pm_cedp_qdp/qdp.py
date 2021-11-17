import itertools
import math
import os
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.transition_system import algorithm as ts_discovery
import pm4py
import pandas as pd
import numpy as np
from pm_cedp_qdp.utilities import utilities
from pathlib import Path
from pm4py.algo.discovery.transition_system import algorithm as ts_discovery
from pm4py.visualization.transition_system import visualizer as ts_vis

class QDP():

    def __init__(self):
        self = self



    def split_log_into_initial_and_continuous(self,log_object,start_point,window_size=1):
        log = pm4py.convert_to_dataframe(log_object)
        log.sort_values('time:timestamp')
        start_df = log[log['concept:name'] == "▶"]
        for index, row in start_df.iterrows():
            log.at[index,'time:timestamp'] = log.iloc[index+1]['time:timestamp']

        end_df = log[log['concept:name'] == "■"]
        for index, row in end_df.iterrows():
            log.at[index, 'time:timestamp'] = log.iloc[index - 1]['time:timestamp']

        log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], utc=True)
        initial_log= log[log['time:timestamp']<=start_point]

        init_log = pm4py.convert_to_event_log(initial_log)
        continuous_log = init_log.__deepcopy__()

        for count,trace in enumerate(init_log):
            len_trace = len(trace._list)
            for i in range(window_size):
                if trace._list[len_trace-1]['concept:name'] != "■":
                    continuous_log[count]._list.append(log_object[count][len_trace])

        return init_log,continuous_log

    def probability_matrices(self, ts):
        DEFAULT_ARTIFICIAL_START_ACTIVITY = "▶"
        DEFAULT_ARTIFICIAL_END_ACTIVITY = "■"
        state_with_id = {}
        for state in ts.states:
            state_with_id[id(state)] = state

        id_states = [id(state) for state in ts.states]
        number_states = len(id_states)
        df_forward = pd.DataFrame(np.zeros((number_states, number_states), dtype=float), index=id_states,
                                  columns=id_states)
        df_backward = pd.DataFrame(np.zeros((number_states, number_states), dtype=float), index=id_states,
                                   columns=id_states)
        for transition in ts.transitions:
            df_forward.at[id(transition.from_state), id(transition.to_state)] = len(
                set(x[0].attributes["concept:name"] for x in transition.data["events"]))
            df_backward.at[id(transition.to_state), id(transition.from_state)] = 1
        for id_state_row in id_states:
            sum_outpout = df_forward.loc[id_state_row].sum()
            sum_outpout_backward = df_backward.loc[id_state_row].sum()
            if sum_outpout == 0:
                df_forward = df_forward.drop(labels=id_state_row, axis=0)
            if sum_outpout_backward == 0:
                df_backward = df_backward.drop(labels=id_state_row, axis=0)
            if sum_outpout == 1:
                continue
            elif sum_outpout != 0:
                df_forward.loc[id_state_row] = [df_forward.at[id_state_row, id_state_column] / sum_outpout for
                                                id_state_column in id_states]

        return df_forward, df_backward, state_with_id

    def probability_matrices_filter(self, df_probability, cont_log, state_with_id):
        df_filtered = pd.DataFrame(data=None, columns=df_probability.columns)
        trace_variant = []
        for trace in cont_log:
            variant = []
            for event in trace:
                variant.append(event['concept:name'])
            trace_variant.append(tuple(variant))
        variants = [list(varianttt) for varianttt in set(trace_variant)]

        for index, row in df_probability.iterrows():
            if state_with_id[index].name in variants:
                df_filtered = df_filtered.append(row)

        return df_filtered

    def commulative_leakage_calc_matrix(self, ts, df_probability, epsilon, previous_alpha):
        # algorithm for FPL/ BPL
        leakage = 0
        rows_number = df_probability.shape[0]
        column_number = df_probability.shape[1]
        list_perms = list(itertools.permutations([i for i in range(rows_number)], 2))
        count = 0
        for perms in list_perms:  # per each 2 rows (q,d) in the probability matrix
            count += 1
            q_plus = list()
            d_plus = list()
            for j in range(column_number):
                if df_probability.iloc[perms[0], j] > df_probability.iloc[perms[1], j]:
                    q_j = df_probability.iloc[perms[0], j].item()
                    d_j = df_probability.iloc[perms[1], j].item()
                    if q_j == 1 and d_j == 0:  # This is based on Theorem 7. It is also obvious based on the equation (24)
                        return previous_alpha + epsilon
                    q_plus.append(q_j)  # append q_j to q_plus
                    d_plus.append(d_j)  # append d_j to d_plus
            q = np.sum(q_plus)
            d = np.sum(d_plus)

            while True and len(q_plus) > 0 and np.sum(d_plus) > 0:
                update = False
                i = 0
                while i < len(q_plus):
                    if d_plus[i] != 0:
                        if q_plus[i] / d_plus[i] <= (q * (math.exp(previous_alpha) - 1) + 1) / (
                                d * (math.exp(previous_alpha) - 1) + 1):
                            q_plus.pop(i)
                            d_plus.pop(i)
                            update = True
                    i += 1
                if not update:
                    break
                else:
                    q = np.sum(q_plus)
                    d = np.sum(d_plus)
            if leakage < math.log((q * (math.exp(previous_alpha) - 1) + 1) / (d * (math.exp(previous_alpha) - 1) + 1)):
                leakage = math.log((q * (math.exp(previous_alpha) - 1) + 1) / (d * (math.exp(previous_alpha) - 1) + 1))
            print(count)
        return leakage + epsilon

    def probability_matrices_non_sparse(self, ts):

        state_with_id = {}
        for state in ts.states:
            state_with_id[id(state)] = state
        id_states = [id(state) for state in ts.states]

        # initialize
        forward_dict = {}
        backward_dict = {}
        for id_state in id_states:
            forward_dict[id_state] = []
            backward_dict[id_state] = []

        for transition in ts.transitions:
            cases_passing_tr = len(set(x[0].attributes["concept:name"] for x in transition.data["events"]))
            forward_dict[id(transition.from_state)].append((id(transition.to_state), cases_passing_tr))
            backward_dict[id(transition.to_state)].append((id(transition.from_state), 1))

        for id_state_row in id_states:
            if len(forward_dict[id_state_row]) < 1:
                del forward_dict[id_state_row]
            else:
                sum_outpout_forward_list = [x[1] for x in forward_dict[id_state_row]]
                sum_outpout_forward = sum(sum_outpout_forward_list)
                if sum_outpout_forward == 1:
                    continue
                else:
                    forward_dict[id_state_row] = [(x[0], x[1] / sum_outpout_forward) for x in
                                                  forward_dict[id_state_row]]
            if len(backward_dict[id_state_row]) < 1:
                del backward_dict[id_state_row]

        return forward_dict, backward_dict, state_with_id

    def probability_dict_filter(self, probability_dict, cont_log, state_with_id, state_size, state_direction):
        trace_variant = []
        for trace in cont_log:
            variant = []
            for event in trace:
                variant.append(event['concept:name'])
            if state_direction == "backward":
                map_variant_to_state_size = variant[-state_size:]
            else:
                map_variant_to_state_size = variant[:state_size]
            trace_variant.append(tuple(map_variant_to_state_size))
        variants = [list(varianttt) for varianttt in set(trace_variant)]

        dict_keys = [key for key in probability_dict]

        for key in dict_keys:
            if state_with_id[key].name not in variants:
                del probability_dict[key]

        return probability_dict

    def comulative_leakage_calc_dict(self, filtered_dict, epsilon, previous_alpha):
        # algorithm for FPL/ BPL
        leakage = 0
        list_perms = list(itertools.permutations([key for key in filtered_dict], 2))
        for perms in list_perms:  # per each 2 rows (q_row,d_row) in the probability matrix
            q_plus = list()
            d_plus = list()
            for q_item in filtered_dict[perms[0]]:
                non_zero_states_in_d = [state_prob[0] for state_prob in filtered_dict[perms[1]]]
                if q_item[1] == 1 and q_item[
                    0] not in non_zero_states_in_d:  # This is based on Theorem 7. It is also obvious based on the equation (24)
                    return previous_alpha + epsilon
                elif q_item[0] in non_zero_states_in_d:
                    d_item = [state_prob for state_prob in filtered_dict[perms[1]] if state_prob[0] == q_item[0]][0]
                    if q_item[1] > d_item[1]:
                        q_plus.append(q_item[1])  # append q_j to q_plus
                        d_plus.append(d_item[1])  # append d_j to d_plus
            q = np.sum(q_plus)
            d = np.sum(d_plus)

            while True and len(q_plus) > 0 and np.sum(d_plus) > 0:
                update = False
                i = 0
                while i < len(q_plus):
                    if d_plus[i] != 0:
                        if q_plus[i] / d_plus[i] <= (q * (math.exp(previous_alpha) - 1) + 1) / (
                                d * (math.exp(previous_alpha) - 1) + 1):
                            q_plus.pop(i)
                            d_plus.pop(i)
                            update = True
                    i += 1
                if not update:
                    break
                else:
                    q = np.sum(q_plus)
                    d = np.sum(d_plus)
            if leakage < math.log((q * (math.exp(previous_alpha) - 1) + 1) / (d * (math.exp(previous_alpha) - 1) + 1)):
                leakage = math.log((q * (math.exp(previous_alpha) - 1) + 1) / (d * (math.exp(previous_alpha) - 1) + 1))
        return leakage + epsilon

    def next_log(self, log, cont_log, window_size=1):
        next_log = cont_log.__deepcopy__()
        for count, trace in enumerate(cont_log):
            len_trace = len(trace._list)
            for i in range(window_size):
                if trace._list[len_trace - 1]['concept:name'] != "■":
                    next_log[count]._list.append(log[count][len_trace])

        return cont_log,next_log

    def recursive_comulative_leakage_calc_dict(self, export, log, cont_log, epsilon, FPL, BPL, only_complete_traces, state_window, state_direction):
        file = open(export, "a")
        util = utilities()
        cont_log, next_log = self.next_log(log, cont_log)

        FPL_valid = False
        if only_complete_traces:
            cont_log_complete = util.remove_incomplete_traces(cont_log)
            if len(cont_log_complete) >= len(cont_log_complete) / 2:
                FPL_valid = True
            ts = ts_discovery.apply(cont_log_complete,
                                    parameters={'direction': state_direction, 'view': "sequence", 'window': state_window,
                                                'include_data': True})
        else:
            FPL_valid = True
            ts = ts_discovery.apply(cont_log,
                                    parameters={'direction': state_direction, 'view': "sequence", 'window': state_window,
                                                'include_data': True})
        forward_dict, backward_dict, state_with_id = self.probability_matrices_non_sparse(ts)
        filtered_forward = self.probability_dict_filter(forward_dict, next_log, state_with_id, state_window, state_direction)
        filtered_backward = self.probability_dict_filter(backward_dict, next_log, state_with_id, state_window, state_direction)
        if FPL_valid:
            FPL = self.comulative_leakage_calc_dict(filtered_forward, epsilon, FPL)
        else:
            FPL = epsilon
        BPL = self.comulative_leakage_calc_dict(filtered_backward, epsilon, BPL)

        TPL = FPL + BPL - epsilon
        line = str(epsilon) + "," + str(FPL) + "," + str(BPL) + "," + str(TPL) + "\n"
        file.write(line)
        file.close()
        print("FPL:" + str(FPL) + " - BPL:" + str(BPL) + " - TPL:" + str(TPL))
        if len([trace for trace in next_log for event in trace]) == len([trace for trace in log for event in trace]):
            return FPL, BPL, TPL
        else:
            return self.recursive_comulative_leakage_calc_dict(export, log, next_log, epsilon, FPL, BPL, only_complete_traces, state_window, state_direction)

    def apply(self, log_name, epsilon, export_csv, recursive = True, only_complete_traces = False, state_window = 200, state_direction = "backward"):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        export = os.path.join(Path(current_dir).parent.absolute(), 'exports', export_csv)
        f = open(export, "w")

        BPL = epsilon
        FPL = epsilon
        TPL = BPL + FPL - epsilon
        line = str(epsilon) + "," + str(FPL) + "," + str(BPL) + "," + str(TPL) + "\n"
        f.write(line)
        print("Initial leakages-->" + "FPL:" + str(FPL) + " - BPL:" + str(BPL) + " - TPL:" + str(TPL))

        log_path = os.path.join(Path(current_dir).parent.absolute(), 'event_logs', log_name)
        log = xes_importer.apply(log_path)

        util = utilities()
        qdp = QDP()

        # -------start_point is the point in time when all the cases started, one can force having complete traces included.-------
        start_point = util.get_start_point(log)
        # start_point = "2011-01-15 16:04:43" #for running_example
        # start_point = "2012-05-07 16:04:43" #for BPIC2013

        # --------adding artificial start ("▶") and end ("■") activities-------
        log = pm4py.insert_artificial_start_end(log)

        # --------initial_log is the first published event log where all the traces started--------
        # --------cont_log is the next pulished event log where one event is published per each previously incomplete published trace----
        initial_log, cont_log = qdp.split_log_into_initial_and_continuous(log, start_point)

        FPL_valid = False
        # -------If you want to only consider the complete traces for generating temporal correlations------
        if only_complete_traces:
            initial_log_complete = util.remove_incomplete_traces(initial_log)
            if len(initial_log_complete) >= len(initial_log) / 2:
                FPL_valid = True
            # ------Generating transition system for calculating temporal correlations-------
            ts = ts_discovery.apply(initial_log_complete,
                                    parameters={'direction': state_direction, 'view': "sequence", 'window': state_window,
                                                'include_data': True})

        else:
            FPL_valid = True
            # ------Generating transition system for calculating temporal correlations-------
            ts = ts_discovery.apply(initial_log,
                                    parameters={'direction': state_direction, 'view': "sequence", 'window': state_window,
                                                'include_data': True})

        # viz = ts_vis.apply(ts, parameters={ts_vis.Variants.VIEW_BASED.value.Parameters.FORMAT: "svg"})
        # ts_vis.view(viz)

        # ------Calculating backward anf forward privacy leakages based on transition system ----------
        forward_dict, backward_dict, state_with_id = qdp.probability_matrices_non_sparse(ts)

        # ------Keeping the probability information of the states that we have in the cont_log---------
        filtered_forward = qdp.probability_dict_filter(forward_dict, cont_log, state_with_id, state_window, state_direction)
        filtered_backward = qdp.probability_dict_filter(backward_dict, cont_log, state_with_id, state_window, state_direction)

        # -----Calculating comulative DP disclosure because of temporal correlations-------
        if FPL_valid:
            FPL = qdp.comulative_leakage_calc_dict(filtered_forward, epsilon, FPL)
        else:
            FPL = epsilon
        BPL = qdp.comulative_leakage_calc_dict(filtered_backward, epsilon, BPL)

        TPL = FPL + BPL - epsilon

        line = str(epsilon) + "," + str(FPL) + "," + str(BPL) + "," + str(TPL) + "\n"
        f.write(line)
        f.close()
        print("FPL:" + str(FPL) + " - BPL:" + str(BPL) + " - TPL:" + str(TPL))

        if recursive:
            FPL, BPL, TPL = qdp.recursive_comulative_leakage_calc_dict(export, log, cont_log, epsilon, FPL, BPL, only_complete_traces, state_window, state_direction)

        return FPL, BPL, TPL

        # -----This implementation is super slow because of sparse matrices-----------
        # df_forward,df_backward,state_with_id = qdp.probability_matrices(ts)
        # filtered_forward = qdp.probability_matrices_filter(df_forward, cont_log, state_with_id)
        # filtered_backward = qdp.probability_matrices_filter(df_backward, cont_log, state_with_id)
        # FPL = qdp.commulative_leakage_calc_matrix(ts,filtered_forward,0.01,0.01)
        # BPL = qdp.commulative_leakage_calc_matrix(ts,filtered_backward,0.01,0.01)

        # -----visualization-------
        # viz = ts_vis.apply(ts, parameters={ts_vis.Variants.VIEW_BASED.value.Parameters.FORMAT: "svg"})
        # ts_vis.view(viz)
