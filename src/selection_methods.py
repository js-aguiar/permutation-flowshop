"""
permutation-flowshop repository

Selection methods for choosing the jobs that will be
removed in the current iteration of Iterated Greedy.
"""

import random
import numpy as np

def roulette_wheel(weights, num_jobs):
    """Apply Fitness Proportionate Selection based on weights.

    Arguments
        weights: numpy array with the weight for each job
        num_jobs: number of jobs to select

    Returns
        chosen: List with the index of selected jobs.
    """
    weights_sum = np.sum(weights)
    chosen = []
    for i in range(num_jobs):
        # Get random value between 0 and total weight
        random_num = random.random() * weights_sum
        # Binary Search
        for k in range(len(weights)):
            random_num -= weights[k]
            if random_num <= 0:
                chosen.append(k+1)
                break
        else:
            # Rounding errors: append last index
            chosen.append(len(weights) - 1)
    return chosen


def tournament_selection(weights, tournament_size, num_jobs):
    """Apply Tournament Selection to select jobs based on weights.

    Arguments
        weights: numpy array with the weight for each job
        tournament_size: number of jobs in each tournament
        num_jobs: number of jobs to select

    Returns
        chosen: List with the index of selected jobs.
    """
    population = [k+1 for k in range(len(weights))]
    chosen = []

    for tournament_count in range(num_jobs):
        # Select tournament participants from general population
        aspirants = random.sample(population, tournament_size)

        # Choose the best in tournament
        best = -1
        for asp in aspirants:
            if weights[asp-1] > best:
                best = weights[asp-1]
                winner = asp

        # Pick random if all jobs have 0 idle time
        if best == 0:
            winner = random.choice(aspirants)

        # Add job to chosen list and remove from population
        chosen.append(winner)
        population.remove(winner)
    return chosen


def stochastic_universal_sampling(weights, num_jobs):
    """Apply Stochastic Universal Sampling to select jobs based on weights.

    Arguments
        weights: numpy array with the weight for each job
        num_jobs: number of jobs to select

    Returns
        chosen: List with the index of selected jobs.
    """
    # Calculate points distance and random start
    distance = np.sum(weights)/num_jobs
    start = random.uniform(0, distance)

    # List of pointers
    points = [start + i*distance for i in range(num_jobs)]
    # Get corresponding jobs
    chosen = []
    for p in points:
        i = 0
        _sum = weights[i]
        while _sum < p:
            i += 1
            _sum += weights[i]
        chosen.append(i+1)
    return chosen
