import copy
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
import time
import random


class QDP():

    def __init__(self):
        self = self

    def split_log_into_initial_and_continuous(self, log_object, start_point, window_size, certain_release):
        log = pm4py.convert_to_dataframe(log_object)
        log.sort_values('time:timestamp')

        #Assigning timestamps to the artificial start and end activitites
        start_df = log[log['concept:name'] == "▶"]
        for index, row in start_df.iterrows():
            log.at[index, 'time:timestamp'] = log.iloc[index + 1]['time:timestamp']
        end_df = log[log['concept:name'] == "■"]
        for index, row in end_df.iterrows():
            log.at[index, 'time:timestamp'] = log.iloc[index - 1]['time:timestamp']


        log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], utc=True)
        initial_log = log[log['time:timestamp'] <= start_point]

        init_log = pm4py.convert_to_event_log(initial_log)



        #-----Statistics-------
        util = utilities()
        con_log = log[log['time:timestamp'] > start_point]
        cont_log = pm4py.convert_to_event_log(con_log)

        complete_traces_in = util.get_number_complete_traces(init_log)
        events_in = util.get_number_events(init_log)
        cases_in = len(init_log)
        incomplete_traces_in = (cases_in - complete_traces_in)
        num_event_in = events_in - (complete_traces_in * 2) - incomplete_traces_in

        complete_traces = util.get_number_complete_traces(cont_log)
        events = util.get_number_events(cont_log)
        cases = len(cont_log)
        incomplete_traces = (cases - complete_traces)
        num_event = events - (complete_traces * 2) - incomplete_traces
        # -----Statistics-------

        init_log, continuous_log = self.next_log(log_object, init_log, window_size, certain_release)

        return init_log, continuous_log

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

    def probability_matrices_non_sparse(self, ts, log, depth, certain_release):

        id2state = {}
        state2id = {}
        id_states = []
        for index, state in enumerate(ts.states):
            if id(state) not in id_states:
                id_states.append(id(state))
                id2state[id(state)] = state
                state2id[state] = id(state)
            else:
                uniq_id = int(time.time()) - index
                id_states.append(uniq_id)
                id2state[uniq_id] = state
                state2id[state] = id(state)

        # id_states = [id(state) for state in ts.states]

        # initialize
        forward_dict = {}
        backward_dict = {}
        for id_state in id_states:
            forward_dict[id_state] = []
            backward_dict[id_state] = []

        for transition in ts.transitions:
            cases_passing_tr = len(set(x[0].attributes["concept:name"] for x in transition.data["events"]))
            forward_dict[state2id[transition.from_state]].append((state2id[transition.to_state], cases_passing_tr))
            backward_dict[state2id[transition.to_state]].append((state2id[transition.from_state], 1))

        #▶
        state_probability = {}
        for state in ts.states:
            prob = len(state.data["ingoing_events"]) / len(log)
            state_probability[state2id[state]] = prob

        for id_state in id_states:
            if len(forward_dict[id_state]) < 1 and '■' not in id2state[id_state].name:
                del forward_dict[id_state]
            if len(backward_dict[id_state]) < 1 and '■' not in id2state[id_state].name:
                del backward_dict[id_state]

        #calcualting the next/previous proabailities
        #if certain release (1 event), then exact states and their probs. Otherwise (0 or 1 event), the probs are divided into remaining at the same state or moving one event)
        self.convert_count2prob(forward_dict, backward_dict, id_states, id2state)

        self.further_explore_forward(forward_dict, depth, id2state, certain_release)
        backward_dict = self.further_explore_backward(backward_dict, depth, certain_release)

        return forward_dict, backward_dict, id2state, state2id

    def convert_count2prob(self, forward_dict, backward_dict, id_states, id2state):
        for id_state_row in id_states:
            if id_state_row in forward_dict:
                if '■' in id2state[id_state_row].name:                      # note that these states does not have any next state before.
                    forward_dict[id_state_row].insert(0,(id_state_row,1))

                sum_outpout_forward_list = [x[1] for x in forward_dict[id_state_row]]
                sum_outpout_forward = sum(sum_outpout_forward_list)
                if sum_outpout_forward == 1:                                # one possible next state will result in probability 1
                    pass
                else:
                    forward_dict[id_state_row] = [(x[0], x[1] / sum_outpout_forward) for x in
                                                  forward_dict[id_state_row]]
                # if '■' not in id2state[id_state_row].name and not certain_release:                      #for non-complete traces, there is a chance equal to their probability to stay unchanged.
                #     forward_dict[id_state_row] = [(x[0], 0.5 * x[1]) for x in forward_dict[id_state_row]]  #if 1 event is published
                #     forward_dict[id_state_row].insert(0, (id_state_row, 0.5 * 1))                           #if 0 events are published
                #     # forward_dict[id_state_row].insert(0, (id_state_row, 0.5 * state_probability[id_state_row]))

            if id_state_row in backward_dict:
                # if not certain_release:
                #     backward_dict[id_state_row].insert(0,(id_state_row, 1))

                sum_outpout_backward_list = [x[1] for x in backward_dict[id_state_row]]
                sum_outpout_backward = sum(sum_outpout_backward_list)
                if sum_outpout_backward == 1:
                    pass
                else:
                    backward_dict[id_state_row] = [(x[0], x[1] / sum_outpout_backward) for x in
                                                   backward_dict[id_state_row]]

    def further_explore_backward(self, backward_dict,depth,certain_release):  # for depth >= 2, this function explores non-local states up to distance 'depth'.
        backward_dict_copy = copy.deepcopy(backward_dict)
        max_backward_move = {}
        for state in backward_dict.keys():
            max_backward_move[state] = 1
            depth_counter = 1
            visited_states = []
            while depth_counter < depth:
                move_backward_counted = False
                depth_counter += 1
                adjacent_states = backward_dict_copy[state].copy()
                adjacent_states_not_visited = list(set([x[0] for x in adjacent_states]) - set(visited_states))
                for adjacent_state in adjacent_states_not_visited:
                    if adjacent_state in backward_dict.keys():
                        one_more_back = (backward_dict[adjacent_state][0][0],backward_dict[adjacent_state][0][1])
                        backward_dict_copy[state].append(one_more_back)
                        if not move_backward_counted:
                            max_backward_move[state] += 1
                            move_backward_counted = True
                        # update all states
                        for index, next_state in enumerate(backward_dict_copy[state]):
                            backward_dict_copy[state][index] = (next_state[0], one_more_back[1])
                        visited_states.append(adjacent_state)

        if not certain_release:
            for id_state in backward_dict_copy.keys():
                backward_dict_copy[id_state].insert(0, (id_state, 1))
                backward_dict_copy[id_state] = [(x[0], 1 / (max_backward_move[id_state] + 1) * x[1]) for x in backward_dict_copy[id_state]]

        return backward_dict_copy

    def further_explore_forward(self, forward_dict, depth, id2state, certain_release):  # for depth >= 2, this function explores non-local states up to distance 'depth'.
        max_forward_move = {}
        for state in forward_dict.keys():
            max_forward_move[state] = 1
            depth_counter = 1
            visited_states = []
            while depth_counter < depth:
                move_forward_counted = False
                adjacent_states = forward_dict[state].copy()
                adjacent_states_not_visited = list(set(adjacent_states) - set(visited_states))
                for adjacent_state in adjacent_states_not_visited:
                    if adjacent_state[0] in forward_dict.keys() and adjacent_state[0] != state:
                        for next_state in forward_dict[adjacent_state[0]]:
                            if next_state[0] != adjacent_state[0]:
                                forward_dict[state].append((next_state[0], next_state[1] * adjacent_state[1]))
                                if not move_forward_counted:
                                    max_forward_move[state] += 1
                                    move_forward_counted = True
                        visited_states.append(adjacent_state)
                depth_counter += 1

        if not certain_release:
            for id_sate in forward_dict.keys():
                if '■' not in id2state[id_sate].name:
                    forward_dict[id_sate] = [(x[0], 1/(max_forward_move[id_sate]+1) * x[1]) for x in forward_dict[id_sate]]
                    forward_dict[id_sate].insert(0, (id_sate, 1/(max_forward_move[id_sate]+1) ))

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

    def next_log(self, log, cont_log, window_size, certain_release):

        next_log = cont_log.__deepcopy__()

        # Events that are needed to be added are based on release scenario and also window_size
        if certain_release:
            event_range = range(window_size)
        else:
            random_int = random.randint(0, window_size)
            event_range = range(random_int)

        exist_cases = [trace.attributes['concept:name'] for trace in cont_log]

        for count_org, trace_org in enumerate(log):
            case_id = trace_org.attributes['concept:name']
            if case_id not in exist_cases:
                trace = log[count_org].__deepcopy__()
                trace._list = []
                for j in event_range:
                    try:
                        trace._list.append(log[count_org][j])
                    except IndexError:
                        break
                next_log._list.append(trace)
                exist_cases.append(case_id)
            else:
                for count_init, trace_init in enumerate(cont_log):
                    if case_id == trace_init.attributes['concept:name']:
                        len_trace = len(trace_init._list)
                        if len_trace == len(log[count_org]):
                            pass
                        else:
                            for i in event_range:
                                if len(log[count_org]) > len_trace + i:
                                    next_log[count_init]._list.append(log[count_org][len_trace + i])
                        break



        # next_log = cont_log.__deepcopy__()
        # for count_org, trace_org in enumerate(log):
        #     case_id = trace_org.attributes['concept:name']
        #     for count_init, trace_init in enumerate(cont_log):
        #         if case_id == trace_init.attributes['concept:name']:
        #             len_trace = len(trace_init._list)
        #             if len_trace == len(log[count_org]):
        #                 pass
        #             else:
        #                 for i in range(window_size):
        #                     if len(log[count_org]) > len_trace + i:
        #                         next_log[count_init]._list.append(log[count_org][len_trace + i])
        #             break


        return cont_log, next_log


    def recursive_comulative_leakage_calc_dict(self, export, log, next_log, epsilon, FPL, BPL, only_complete_traces,
                                               state_window, state_direction, explore_depth, window_size,
                                               certain_release, release_counter, release_index, BPL_list, FPL_list, TPL_list, max_num_release):

        file = open(export, "a")
        util = utilities()
        current_log, next_log = self.next_log(log, next_log, window_size, certain_release)

        FPL_valid = False
        if only_complete_traces:
            last_log_complete = util.remove_incomplete_traces(next_log)
            if len(last_log_complete) >= len(next_log) / 2:
                FPL_valid = True
            ts = ts_discovery.apply(last_log_complete,
                                    parameters={'direction': state_direction, 'view': "sequence", 'window': state_window,
                                                'include_data': True})
        else:
            FPL_valid = True
            ts = ts_discovery.apply(next_log,
                                    parameters={'direction': state_direction, 'view': "sequence", 'window': state_window,
                                                'include_data': True})
        forward_dict, backward_dict, id2state, state2id = self.probability_matrices_non_sparse(ts, next_log, explore_depth, certain_release)
        # filtered_forward = self.probability_dict_filter(forward_dict, next_log, id2state, state_window, state_direction)
        # filtered_backward = self.probability_dict_filter(backward_dict, next_log, id2state, state_window, state_direction)
        if FPL_valid:
            FPL = self.comulative_leakage_calc_dict(forward_dict, epsilon, FPL)
        else:
            FPL = epsilon
        BPL = self.comulative_leakage_calc_dict(backward_dict, epsilon, BPL)

        TPL = FPL + BPL - epsilon
        release_counter += 1
        release_index.append(release_counter)
        BPL_list.append(BPL)
        FPL_list.append(FPL)
        TPL_list.append(TPL)

        line = str(epsilon) + "," + str(FPL) + "," + str(BPL) + "," + str(TPL) + "\n"
        file.write(line)
        file.close()
        print("FPL:" + str(FPL) + " - BPL:" + str(BPL) + " - TPL:" + str(TPL))
        if len([trace for trace in next_log for event in trace]) == len([trace for trace in log for event in trace]) or release_counter == max_num_release:
            return release_index, BPL_list, FPL_list, TPL_list
        else:
            return self.recursive_comulative_leakage_calc_dict(export, log, next_log, epsilon, FPL, BPL,
                                                               only_complete_traces, state_window, state_direction,
                                                               explore_depth, window_size, certain_release,
                                                               release_counter, release_index, BPL_list, FPL_list, TPL_list, max_num_release)


    def apply(self, log_name, epsilon, export_csv, recursive=True, only_complete_traces=False, state_window=200,
              state_direction="backward", explore_depth=1, window_size=1, certain_release= True, event_percentage = 0.5, max_num_release = 10):

        release_counter = 1; release_index = [1]; BPL_list = [epsilon]; FPL_list = [epsilon]; TPL_list = [epsilon]

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
        variant = xes_importer.Variants.ITERPARSE
        parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
        log = xes_importer.apply(log_path,
                                 variant=variant, parameters=parameters)
        util = utilities()
        qdp = QDP()

        all_cases = False #all cases have to be started in the first release
        force_one_complete_trace = False #the first release is forced to have one complete trace
        start_point = util.get_start_point(log, event_percentage=event_percentage, all_cases = all_cases, force_one_complete_trace = force_one_complete_trace)

        # --------adding artificial start ("▶") and end ("■") activities-------
        log = pm4py.insert_artificial_start_end(log)

        # --------initial_log is the first published event log where all the traces started--------
        # --------cont_log is the next pulished event log where one event is published per each previously incomplete published trace----
        initial_log, next_log = qdp.split_log_into_initial_and_continuous(log, start_point, window_size, certain_release)

        FPL_valid = False
        # -------If you want to only consider the complete traces for generating temporal correlations------
        if only_complete_traces:
            last_log_complete = util.remove_incomplete_traces(next_log)
            if len(last_log_complete) >= len(next_log) / 2:
                FPL_valid = True
            # ------Generating transition system for calculating temporal correlations-------
            ts = ts_discovery.apply(last_log_complete,
                                    parameters={'direction': state_direction, 'view': "sequence", 'window': state_window,
                                                'include_data': True})

        else:
            FPL_valid = True
            # ------Generating transition system for calculating temporal correlations-------
            ts = ts_discovery.apply(next_log,
                                    parameters={'direction': state_direction, 'view': "sequence", 'window': state_window,
                                                'include_data': True})

        # viz = ts_vis.apply(ts, parameters={ts_vis.Variants.VIEW_BASED.value.Parameters.FORMAT: "svg"})
        # ts_vis.view(viz)

        # ------Calculating backward anf forward privacy leakages based on transition system ----------
        forward_dict, backward_dict, id2state, state2id = qdp.probability_matrices_non_sparse(ts, next_log, explore_depth, certain_release)

        # ------Keeping the probability information of the states that we have in the next_log---------
        # filtered_forward = qdp.probability_dict_filter(forward_dict, next_log, id2state, state_window, state_direction)
        # filtered_backward = qdp.probability_dict_filter(backward_dict, next_log, id2state, state_window, state_direction)

        for key,value in forward_dict.items():
            if len(value) == 1:
                here = value

        # -----Calculating comulative DP disclosure because of temporal correlations-------
        if FPL_valid:
            FPL = qdp.comulative_leakage_calc_dict(forward_dict, epsilon, FPL)
        else:
            FPL = epsilon
        BPL = qdp.comulative_leakage_calc_dict(backward_dict, epsilon, BPL)
        TPL = FPL + BPL - epsilon

        release_counter += 1
        release_index.append(release_counter)
        BPL_list.append(BPL); FPL_list.append(FPL); TPL_list.append(TPL)
        line = str(epsilon) + "," + str(FPL) + "," + str(BPL) + "," + str(TPL) + "\n"
        f.write(line)
        f.close()
        print("FPL:" + str(FPL) + " - BPL:" + str(BPL) + " - TPL:" + str(TPL))

        if recursive:
            release_index, BPL_list, FPL_list, TPL_list = qdp.recursive_comulative_leakage_calc_dict(export, log, next_log, epsilon, FPL, BPL,
                                                                       only_complete_traces, state_window, state_direction,
                                                                       explore_depth, window_size, certain_release,
                                                                       release_counter, release_index, BPL_list, FPL_list, TPL_list, max_num_release)

        return release_index, BPL_list, FPL_list, TPL_list

