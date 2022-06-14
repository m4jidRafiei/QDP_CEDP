import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class utilities():

    def __init__(self):
        self = self

    def get_smallest_complete(self,log):
        complete_time_list = [trace[-1]["time:timestamp"] for trace in log]
        smallest_complete = min(complete_time_list)
        return smallest_complete

    def get_number_events(self,log):
        event_num = 0
        for trace in log:
            for event in trace:
                event_num += 1
        return event_num

    def get_start_point(self, log, event_percentage = 0.5, all_cases = False, force_one_complete_trace = False):

        start_point = 0

        event_num = self.get_number_events(log)
        count = 0
        for trace in log:
            for event in trace:
                count += 1
                if count == int(event_percentage * event_num) + 1:
                    start_point = event["time:timestamp"]

        if all_cases:
            latest_case = max([trace[0]["time:timestamp"] for trace in log])
            if start_point < latest_case:
                start_point = latest_case

        if force_one_complete_trace:
            smallest_complete = self.get_smallest_complete(log)
            if start_point < smallest_complete:
                start_point = smallest_complete

        return start_point


    def get_number_complete_traces(self,log):
        # last_activities = [trace[-1]["concept:name"] for trace in log]
        counter = 0
        for trace in log:
            if trace[0]["concept:name"] == "▶" and trace[-1]["concept:name"] == "■":
                counter +=1
        return counter


    def remove_incomplete_traces(self, log):
        remove_indeces = []
        for index, trace in enumerate(log):
            len_trace = len(trace._list)
            if trace._list[len_trace - 1]['concept:name'] != "■":
                remove_indeces.append(index)

        initial_log = [i for j, i in enumerate(log) if j not in remove_indeces]

        return initial_log

    def draw_plot(self, release_index, BPL_list, FPL_list, TPL_list, export_jpg):
        plt.rcParams["figure.figsize"] = (6, 5)
        fig, ax = plt.subplots()
        ax.set_xticks(release_index)
        plt.plot(release_index, BPL_list, color='blue', markersize=14, marker='o', label='BPL')
        plt.plot(release_index, FPL_list, color='orange', marker='s', markersize=9, label='FPL')
        plt.plot(release_index, TPL_list, color='gray', marker='^', markersize=7, label='TPL')
        # plt.title('PL in CEDP', fontsize=14)
        plt.xlabel('Event Log Release', fontsize=12)
        plt.ylabel('Privacy Leakage', fontsize=12)
        legend = plt.legend(loc='upper left', shadow=True, fontsize='large')
        plt.grid(True, axis='y')
        plt.savefig(export_jpg, dpi=300)
        plt.show()