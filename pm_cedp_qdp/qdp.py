import itertools
import math
from pm4py.algo.discovery.transition_system import algorithm as ts_discovery
import pm4py
import pandas as pd
import numpy as np
from pm_cedp_qdp.utilities import utilities

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

    def probability_dict_filter(self, probability_dict, cont_log, state_with_id):
        trace_variant = []
        for trace in cont_log:
            variant = []
            for event in trace:
                variant.append(event['concept:name'])
            trace_variant.append(tuple(variant))
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

    def recursive_comulative_leakage_calc_dict(self, log, cont_log, epsilon, FPL, BPL):
        util = utilities()
        cont_log, next_log = self.next_log(log, cont_log)
        cont_log_complete = util.remove_incomplete_traces(cont_log)

        FPL_valid = False
        if len(cont_log_complete) >= len(cont_log_complete) / 2:
            FPL_valid = True

        ts = ts_discovery.apply(cont_log_complete,
                                parameters={'direction': "backward", 'view': "sequence", 'window': 200,
                                            'include_data': True})
        forward_dict, backward_dict, state_with_id = self.probability_matrices_non_sparse(ts)
        filtered_forward = self.probability_dict_filter(forward_dict, next_log, state_with_id)
        filtered_backward = self.probability_dict_filter(backward_dict, next_log, state_with_id)
        if FPL_valid:
            FPL = self.comulative_leakage_calc_dict(filtered_forward, epsilon, FPL)
        else:
            FPL = epsilon
        BPL = self.comulative_leakage_calc_dict(filtered_backward, epsilon, BPL)

        TPL = FPL + BPL - epsilon
        print("FPL: " + str(FPL))
        print("BPL: " + str(BPL))
        print("TPL: " + str(TPL))
        if len([trace for trace in next_log for event in trace]) == len([trace for trace in log for event in trace]):
            return FPL, BPL, TPL
        else:
            return self.recursive_comulative_leakage_calc_dict(log, next_log, epsilon, FPL, BPL)
