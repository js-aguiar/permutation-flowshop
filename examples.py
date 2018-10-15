"""
permutation-flowshop repository

This module has two examples on how to setup and run
the algorithm for Permutation Flowshop scheduling problems.
The first one uses random generated data, while the second
uses one of the instances from the Taillard benchmark set.
"""

import numpy as np
import benchmark
from iterated_greedy import IteratedGreedy


def example1_random_data():
    """Execute the algorithm with randomly generated data."""

    # Generate a (20, 5) array with integer numbers
    rnd_data = np.random.randint(size=(20,5), low=5, high=80)
    print(rnd_data)

    ig = IteratedGreedy(rnd_data)   # Create problem instance
    ig.run(5000)   # Run the default algorithm for 5000ms (5 seconds)

    # Print results to console
    print("Best makespan", ig.best_solution.makespan,"iterations:", ig.iterations)
    print("Job sequence:", ig.best_solution.sequence)


def example2_taillard():
    """Execute the algorithm with the first Taillard instance."""

    # Load instance
    instance = benchmark.import_taillard()[0]
    print(instance)

    ig = IteratedGreedy(instance)
    # Run the algorithm with local search on partial solutions
    # and removing 5 jobs at each iteration
    ig.local_search_partial_solution = True
    ig.num_jobs_remove = 5
    ig.run(10000)

    # Print results to console
    print("Best makespan", ig.best_solution.makespan,"iterations:", ig.iterations)
    print("Job sequence:", ig.best_solution.sequence)


if __name__ == "__main__":
    example1_random_data()
    example2_taillard()