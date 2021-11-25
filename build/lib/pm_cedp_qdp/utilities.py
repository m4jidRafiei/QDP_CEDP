
class utilities():

    def __init__(self):
        self = self

    def get_smallest_complete(self,log):
        smallest_complete = min(trace[-1]["time:timestamp"] for trace in log)
        return smallest_complete

    def get_start_point(self, log, force_one_complete_trace = False):
        start_point = max([trace[0]["time:timestamp"] for trace in log])
        if force_one_complete_trace:
            smallest_complete = self.get_smallest_complete(log)
            if start_point < smallest_complete:
                start_point = smallest_complete
        return start_point

    def remove_incomplete_traces(self, log):
        remove_indeces = []
        for index, trace in enumerate(log):
            len_trace = len(trace._list)
            if trace._list[len_trace - 1]['concept:name'] != "■":
                remove_indeces.append(index)

        initial_log = [i for j, i in enumerate(log) if j not in remove_indeces]

        return initial_log