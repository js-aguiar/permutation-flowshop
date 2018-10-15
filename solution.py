"""
permutation-flowshop repository

Class for holding solutions and connecting makespan calculations
that are implemented in C (through Cython library).
"""

import numpy as np
import calculations

class Solution(object):
    """Implements functions and data structures for the problem and solution.

    Hold the processing times for the problem as well as the solution sequence,
    makesppan and idle time. Also implement functions for calculating completion
    times, makespan, best insert position and idle time. Calculations are done
    in the compiled Cython module. Makespan and idle time must be integers.

    Attributes:
        num_jobs: Number of jobs to be sequenced (int).
        num_machines: Number of machines in the problem (int).
        makespan: Current makespan of the sequence (int, default: 0).
        idle_time: Current idle time between jobs (int, default: 0).
        sequence: List with the current sequence of jobs
        idle_times: Numpy array with the idle time associated to each job.
    """

    def __init__(self, instance_processing_times):
        # int variables
        self.num_jobs = len(instance_processing_times)
        self.num_machines = len(instance_processing_times[0])
        self.makespan = 0
        self.idle_time = 0

        # Current solution sequence - list object
        self.sequence = list()
        # Processing times - numpy 2d array
        self.processing_times = instance_processing_times
        # Idle times - numpy array
        self.idle_times = np.zeros(shape=(self.num_jobs), dtype= 'int32')


    def calculate_completion_times(self):
        """Calculate completion time."""
        sequence_np = np.array(self.sequence, dtype='int32')
        memory_view_object = calculations.calculate_completion_times(sequence_np, self.processing_times, self.num_machines, 1)
        return np.array(memory_view_object)


    def calculate_makespan(self):
        """Calculate makespan for the sequence."""
        sequence_np = np.array(self.sequence, dtype='int32')
        self.makespan = calculations.calculate_completion_times(sequence_np,self.processing_times,self.num_machines, 0)
        return self.makespan


    def insert_best_position(self, job, tie_breaking=False):
        """ Insert the given job in the position that minimize makespan.

        Insert the job in the position at self.sequence that minimizes the
        sequence makespan.

        Arguments
            job: Job to be inserted (int).
            tie_breaking: Use tie breaking mechanism (boolean, default: False).

        Returns
            makespan: Makespan after inserting the job.
        """
        if tie_breaking:
            use_tie_breaking = 1
        else:
            use_tie_breaking = 0

        sequence_np = np.array(self.sequence, dtype='int32')
        best_position, self.makespan = calculations.taillard_acceleration\
            (sequence_np,self.processing_times,job,self.num_machines,use_tie_breaking)

        self.sequence.insert(best_position - 1, job)
        return self.makespan


    def calculate_idle_times(self):
        """Calculate the idle time wrt each job and saves in self.idle_time."""
        sequence_np = np.array(self.sequence, dtype='int32')
        memory_view_object = calculations.calculate_idle_times(sequence_np,self.processing_times,self.num_machines)
        self.idle_time = np.array(memory_view_object)