## Iterated Greedy for the Permutation Flowshop Scheduling Problem

Scheduling problems are a class of optimization problem in computer science and operations research in which jobs are assigned to resources at particular times.
Each job has a specific processing time in each resource or machine and the goal is to find the processing sequence that minimizes a given metric.

The permutation flowshop is a specific case of scheduling problems in which all machines must have the same jobs sequence. 
Therefore, we have a unique sequence that must have all jobs and can be considered a vector of size n_jobs.
This is considered an *NP-Hard* problem and is usually solved with specialized heuristics, 
specially for larger instances where the number of combinations is too large for classic optimization techniques.

This repository is a Python implementation of the Iterated Greedy metaheuristic for the PFSP problem. In this algorithm, k jobs are removed
from the sequence and inserted one by one in the position with minimal makespan. The new sequence can be accepted or rejected according
to the acceptance rule, which is based on the Simulated Annealing heuristic. The most computacional expensive parts were implemented
in C trough the Cython library (calculations.pyx file).

The Iterated Greedy was first proposed by Ruiz and Stutzle (2007) to this problem, but a few variations appeared later and were also
implemented through hyperparameters for the algorithm:

1. Ruiz, Stützle. A simple and effective iterated greedy algorithm for the
permutation flowshop scheduling problem". EJOR v177(3), p2033–2049, 2007. Original implementation - use the default hyperparameters.

2. Fernandez-Viagas, Framinan. On insertion tie-breaking rules in heuristics for the permutation
flowshop scheduling problem. Computers and Operations Research, v45, p60–67, 2014. Add a tie breaking mechanism based on idle times 
when two insertion positions have the same makespan - set the tie_breaking parameter to True.

3. Dubois-Lacoste, Pagnozzi, Stützle. An iterated greedy algorithm with optimization of partial solutions for
the makespan permutation flowshop problem. Computers and Operations Research, v81, p160–166, 2017. Add local search
on partial solutions (after jobs are removed) - set the local_search_partial_solution to True.

## Installation

The following instructions are for Linux based machine. For Windows see [this tutorial](https://github.com/cython/cython/wiki/InstallingOnWindows).

1. Make sure Python 3.5 is installed

2. Install numpy, pandas and cython packages

3. Build the C code with the command:

```python3 cython_setup.py build_ext --inplace```

4. Check the examples.py file to see how to set your problem
