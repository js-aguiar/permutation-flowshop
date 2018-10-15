"""
permutation-flowshop repository

Module that implements constructive heuristics for the
flowshop scheduling problem.
"""

import random
import numpy as np

def NEH(solution, tie_breaking=False, order_jobs="SD"):
    """Create initial solution with NEH heuristic.

    Apply the Nawaz, Enscore and Hans heuristic (1983) to the
    solution object argument.

    Arguments:
        solution: Solution object.
        tie_breaking: Use tie breaking mechanism (boolean, default: False).
        order_jobs: How to order jobs, possible values are:
        SD - Non decreasing sum of processing times (original order)
        AD - Non decreasing sum of the mean and deviation of processing times
        RD - Jobs are randomly ordered
    """
    # Order jobs
    if order_jobs == 0:
        sorted_jobs = _sd_order(solution)
    elif order_jobs == 1:
        sorted_jobs = _ad_order(solution)
    else:
        sorted_jobs = [i for i in range(1, solution.num_jobs + 1)]
        random.shuffle(sorted_jobs)
    # Take the first two jobs and schedule them in order to minimize the partial makespan
    solution.sequence = [sorted_jobs[0],sorted_jobs[1]]
    makespan1 = solution.calculate_makespan()
    solution.sequence = [sorted_jobs[1], sorted_jobs[0]]
    if makespan1 < solution.calculate_makespan():
        solution.sequence = [sorted_jobs[0],sorted_jobs[1]]
        solution.makespan = makespan1
    # For i = 3 to n: Insert the i-th job at the place, among
    # the i possible ones, which minimize the partial makespan
    for job in sorted_jobs[2:]:
        solution.insert_best_position(job, tie_breaking)


def _sd_order(solution):
    """Order jobs by non decreasing sum of the processing times."""
    total_processing_times = dict()
    for i in range(1, solution.num_jobs + 1):
        total_processing_times[i] = np.sum(solution.processing_times[i-1])

    return sorted(total_processing_times, key=total_processing_times.get, reverse=True)


def _ad_order(solution):
    """Order jobs by non-decreasing sum of the mean and deviation (Huang and Chen, 2008)."""
    average_plus_deviation = dict()
    for i in range(1, solution.num_jobs + 1):
        avg = np.mean(solution.processing_times[i-1])
        dev = np.std(solution.processing_times[i-1])
        average_plus_deviation[i] = avg + dev
    return sorted(average_plus_deviation, key=average_plus_deviation.get, reverse=True)