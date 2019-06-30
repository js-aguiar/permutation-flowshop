"""
permutation-flowshop repository

Main class for running the algorithm.
"""

import math
import random
import numpy as np
from datetime import datetime, timedelta
from solution import Solution
import local_search
import constructive_heuristic
import selection_methods

class IteratedGreedy(object):
    """Iterated Greedy Metaheuristic for the PFSP with makespan objective.

    Class that implements the Iterated Greedy algorithm for the
    permutation flowshop problem with the objective of minimizing
    the makespan of the sequence. In the permutation problem, all
    machines must have the same job processing sequence.

    This class also implements a few variations of the original
    algorithm through the following parameters:

    Attributes:
        temperature_param: Temperature parameter (float, default: 0.4)
        num_jobs_remove: Number of jobs to remove at each iteration (int, default: 4)
        tie_breaking: Use tie breaking mechanism (boolean, default: False).
        local_optimum: Repeat local search until local optimum (boolean, default: True).
        tournament_size: Tournament size if selection method is tournament (int, default: 0).
        local_search_partial_solution: Apply local search after removing
        each job (boolean, default: False)

        neh_order_jobs: How to order jobs in the NEH algorithm;
        Zero is for sd, 1 ad, 2 random (int, default: 0).

        selection_method: How to select removed jobs;
        zero for random, one for tournament selection, two
        for fitness proportionate selection and three for
        stochastic universal sampling (int, default: 0)
    """

    def __init__(self, instance_processing_times):
        # Create Solution object
        self.current_solution = Solution(instance_processing_times)
        self.new_solution = Solution(instance_processing_times)
        self.best_solution = Solution(instance_processing_times)

        # Standard algorithm params
        self.temperature_param = 0.4
        self.num_jobs_remove = 4
        self.neh_order_jobs = 0
        self.tie_breaking = False
        self.local_optimum = True
        self.local_search_partial_solution = False
        self.selection_method = 0
        self.tournament_size = 5

    def run(self, runtime_in_miliseconds):
        """Run the Iterated Greedy algorithm.

        Arguments:
            runtime_in_miliseconds: Time to run the algorithm in miliseconds.
        """
        # 0) Define constant temperature and run time
        self.iterations = 0
        temperature = self._calculate_temperature()
        time_limit = datetime.now() + timedelta(milliseconds=runtime_in_miliseconds)

        # 1) First solution (NEH Heuristic + Local Search)
        constructive_heuristic.NEH(self.current_solution, self.tie_breaking, self.neh_order_jobs)
        local_search.insertion_neighborhood(self.current_solution, self.local_optimum, self.tie_breaking)

        # Save best solution and makespan
        self.best_solution.sequence = self.current_solution.sequence.copy()
        self.best_solution.makespan = self.current_solution.makespan

        while datetime.now() < time_limit:

            # 2) Destruction phase - create new solution without a group of jobs
            removed_jobs = self._select_jobs_to_remove()
            self.new_solution.sequence = [job for job in self.current_solution.sequence if job not in removed_jobs]

            # 2.1) Local search on partial solution (optional)
            if self.local_search_partial_solution:
                local_search.insertion_neighborhood(self.new_solution, self.local_optimum, self.tie_breaking)

            # 3) Construction phase
            for job in removed_jobs:
                # Insert in best position (also calculate makespan)
                self.new_solution.insert_best_position(job, self.tie_breaking)

            # 4) Local search on complete solution
            local_search.insertion_neighborhood(self.new_solution, self.local_optimum, self.tie_breaking)

            # 5) Acceptance Criteria
            if self.new_solution.makespan < self.current_solution.makespan:
                # Accept new solution
                self.current_solution.sequence = self.new_solution.sequence.copy()
                self.current_solution.makespan = self.new_solution.makespan

                # Check if best solution
                if self.current_solution.makespan < self.best_solution.makespan:
                    self.best_solution.makespan = self.current_solution.makespan
                    self.best_solution.sequence = self.current_solution.sequence.copy()

            else:

                # Metropolis acceptance criterion - Osman and Potts (1989)
                diff = self.new_solution.makespan - self.current_solution.makespan
                acceptance_criterion = math.exp(- diff / temperature)

                if random.random() <= acceptance_criterion:
                    # Accept new solution
                    self.current_solution.sequence = self.new_solution.sequence.copy()
                    self.current_solution.makespan = self.new_solution.makespan

            self.iterations += 1

    def computational_time(self, runtime_parameter):
        """Return the runtime according to the number of jobs, machines and argument.

        Return the runtime in miliseconds proposed in the literature, which
        depends on an arbitrary parameter and the number of jobs and machines.

        Arguments:
            runtime_parameter: Usually this param is 30, 60, 90 or 120 (int).
        """
        tmp = self.current_solution.num_jobs * (self.current_solution.num_machines / 2)
        return tmp * runtime_parameter

    def _calculate_temperature(self):
        """Return the temperature for acceptance criteria."""
        temperature = 0
        for i in range(self.current_solution.num_jobs):
            temperature += np.sum(self.current_solution.processing_times[i])

        div = self.current_solution.num_jobs * self.current_solution.num_machines * 10
        return self.temperature_param * (temperature / div)



    def _select_jobs_to_remove(self):
        """Return the list of jobs that will be removed in the current iteration."""
        if self.selection_method == 1:
            self.current_solution.calculate_idle_times()
            weights = self.current_solution.idle_time
            weights[self.current_solution.sequence[0] - 1] = 0
            selected_jobs = selection_methods.tournament_selection(weights, self.tournament_size, self.num_jobs_remove)
        elif self.selection_method == 2:
            self.current_solution.calculate_idle_times()
            weights = self._selection_weight_function()
            weights[self.current_solution.sequence[0] - 1] = 0
            selected_jobs = selection_methods.roulette_wheel(weights, self.num_jobs_remove)
        elif self.selection_method == 3:
            self.current_solution.calculate_idle_times()
            weights = self._selection_weight_function()
            weights[self.current_solution.sequence[0] - 1] = 0
            selected_jobs = selection_methods.stochastic_universal_sampling(weights, self.num_jobs_remove)
        else:
            selected_jobs = random.sample(self.current_solution.sequence, self.num_jobs_remove)
        return selected_jobs

    def _selection_weight_function(self):
        """Calculate the weight for each job (wrt the probability of being removed)."""
        # weight for job i: (idle_time + processing_time)/processing_time
        weights = np.empty(self.current_solution.num_jobs)
        for i in range(self.current_solution.num_jobs):
            _sum = np.sum(self.current_solution.processing_times[i])
            weights[i] = (_sum + self.current_solution.idle_time[i])/_sum
        return weights