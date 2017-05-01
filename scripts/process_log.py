#!/usr/bin/env python
"""A script to analyzes the controller log file.
"""

import sys

class ApplicationStat(object):
    """Stores the running statistics of an application.
    """
    def __init__(self, name, parameter):
        self.name = name
        self.parameter = parameter
        self.cycle_stat = dict()
        self.statement_stat = dict()
        self.partition_stat = dict()

    def add_cycle_stat(self, stage_id, statement_id, cycles):
        stage_stat = self.cycle_stat.setdefault(statement_id, dict())
        stage_stat[stage_id] = stage_stat.get(stage_id, 0) + cycles

    def add_statement_stat(self, stage_id, statement_id, timestamp):
        stage_stat = self.statement_stat.setdefault(statement_id, dict())
        assert stage_id not in stage_stat
        stage_stat[stage_id] = timestamp

    def add_partition_stat(self, variable_group_id, partition_id):
        variable_stat = self.partition_stat.setdefault(variable_group_id, set())
        variable_stat.add(partition_id)

    def print_summary(self):
        print 'name = {}'.format(self.name)
        print 'parameter = {}'.format(self.parameter)
        num_partition = max(len(partitions)
                            for partitions in self.partition_stat.values())
        print 'partitioning = {}'.format(num_partition)
        for statement_id, stage_cycles in self.cycle_stat.items():
            if len(stage_cycles) < 3:
                continue
            # Throws away the first and the last iteration.
            sample_stages = sorted(stage_cycles)[1:-1]
            print 'total cycles per iteration (statement {}) = {}'.format(
                statement_id,
                sum(stage_cycles[stage_id] for stage_id in sample_stages)
                / len(sample_stages))
        for statement_id, stage_timestamps in self.statement_stat.items():
            # Throws away the first and the last iteration.
            sample_stages = sorted(stage_timestamps)[1:-1]
            intervals = [stage_timestamps[sample_stages[i+1]] -
                         stage_timestamps[sample_stages[i]]
                         for i in range(0, len(sample_stages) - 1)]
            intervals = sorted(intervals)
            print 'interval (statement {}) min={} max={} 2min={} 2max={} mean={} medium={}'.format(
                statement_id, min(intervals), max(intervals),
                intervals[2], intervals[-3],
                sum(intervals) / len(intervals),
                intervals[len(intervals)/2])

def analyze_file(filename):
    """Analyzes a log file.
    """
    application_stat_dict = dict()
    worker_stat = set()
    with open(filename, 'r') as file_handle:
        for line in file_handle:
            category = line[0]
            if category == 'L':
                application_stat_dict[int(line.split()[1])] = \
                        ApplicationStat(line.split()[2], line.split()[3])
            elif category == 'P':
                application_id = int(line.split()[1])
                application_stat = application_stat_dict[application_id]
                application_stat.add_partition_stat(int(line.split()[2]),
                                                    int(line.split()[3]))
                worker_stat.add(int(line.split()[5]))
            elif category == 'C':
                stage_id = int(line.split()[1])
                statement_id = int(line.split()[2])
                cycles = float(line.split()[3])
                application_stat.add_cycle_stat(stage_id, statement_id, cycles)
            elif category == 'T':
                stage_id = int(line.split()[1])
                statement_id = int(line.split()[2])
                timestamp = float(line.split()[3])
                application_stat.add_statement_stat(stage_id,
                                                    statement_id, timestamp)
            elif category == 'B':
                application_stat_dict = dict()
                worker_stat = set()
    print 'worker = {}'.format(len(worker_stat))
    for application_stat in application_stat_dict.values():
        print
        application_stat.print_summary()

if __name__ == '__main__':
    analyze_file(sys.argv[1])
