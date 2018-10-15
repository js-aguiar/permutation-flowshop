"""
permutation-flowshop repository

Algorithms for local search.
"""

from random import shuffle

def insertion_neighborhood(solution, local_optimum=True, tie_breaking=False):
    """Insertion neighborhood local search algorithm.

    Arguments:
        solution: Solution object (class from solution module).
        local_optimum: Repeat search until local opt (boolean, default: True).
        tie_breaking: Use tie breaking mechanism (boolean, default: False).
    """
    current_makespan = 0
    improve = True

    # If parameter local_optimum is true: repeat while solution is improved
    while improve:
        improve = False
        # Select random elements without repetition - O(N) implementation
        not_tested = solution.sequence.copy()
        shuffle(not_tested)
        for removed_job in not_tested:
            solution.sequence.remove(removed_job)
            # If first iteration then calculate makespan
            if current_makespan == 0:
                current_makespan = solution.calculate_makespan()
            # Insert job in best position and check improvement
            solution.insert_best_position(removed_job, tie_breaking)
            if solution.makespan < current_makespan:
                improve = True
                current_makespan = solution.makespan
        if local_optimum == False:
            break