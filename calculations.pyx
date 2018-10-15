
"""
permutation-flowshop repository

Cython implementation of computer time expensive functions.
"""

from numpy import zeros
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef taillard_acceleration(int[:] sequence, int[:,:] processing_times, int inserting_job, int num_machines, int use_tie_breaking):
	"""Find the best position to insert a job (lowest makespan time).

	Reference:
	"Some efficient heuristic methods for the flow shop sequencing problem",
	Taillard, E., EJOR 47 (1990), p65-74.

	Arguments:
		sequence: Numpy array with current solution (job sequence).
		processing_times: Numpy 2d array with processing times.
		inserting_job: Job to insert (int)
		num_machines: Number of machines in this problem (int).
		use_tie_breaking: Use tie breaking mechanism (int, 1 or 0)

	Returns:
		best_position: Index for position with min makespan.
		best_makespan: Makespan after inserting the job
	"""
	# Static C arrays
	# Faster than using memory view; could also use malloc
	# for dynamic memory allocation
	cdef int e[801][61]
	cdef int q[801][61]
	cdef int f[801][61]
	cdef int ms[801]
	cdef int sequence_length, best_makespan, best_position
	cdef int i, j, iq, jq, tmp

	# Initialize some values
	sequence_length = len(sequence)
	iq = sequence_length + 1

	# Main Loop
	for i in range(1, sequence_length + 2):
		if i < sequence_length + 1:
			e[i][0] = 0

			# Q index I
			iq = iq - 1
			q[iq][num_machines + 1] = 0

		f[i][0] = 0
		jq = num_machines + 1
	
		for j in range(1, num_machines + 1):
			if i == 1:
				e[0][j] = 0
				q[sequence_length + 1][num_machines + 1 - j] = 0
			if i < sequence_length + 1:
				# Q Index J
				jq = jq - 1

				if e[i][j - 1] > e[i - 1][j]:
					e[i][j] = e[i][j - 1] + processing_times[sequence[i - 1]-1, j-1]
				else:
					e[i][j] = e[i - 1][j] + processing_times[sequence[i - 1]-1, j-1]

				if q[iq][jq + 1] > q[iq + 1][jq]:
					q[iq][jq] = q[iq][jq + 1] + processing_times[sequence[iq - 1]-1, jq-1]
				else:
					q[iq][jq] = q[iq + 1][jq] + processing_times[sequence[iq - 1]-1, jq-1]

			# f(ij) = max {f(i, j-1), e(i-1, j)}
			if f[i][j - 1] > e[i - 1][j]:
				f[i][j] = f[i][j - 1] + processing_times[inserting_job - 1, j-1]
			else:
				f[i][j] = e[i - 1][j] + processing_times[inserting_job - 1, j-1]

    # Makespam - job k in position i
    # Also save the first position with the optimal makespan
	best_makespan = 0
	best_position = 0
	for i in range(1, sequence_length + 2):
		ms[i] = 0
		for j in range(1, num_machines + 1):
			tmp = f[i][j] +	q[i][j]
			if tmp > ms[i]:
				ms[i] = tmp
	    # Check best insertion position
		if ms[i] < best_makespan or best_makespan == 0:
			best_makespan = ms[i]
			best_position = i

	# Use tie breaking mechanism (TBFF)
	if use_tie_breaking > 0:
		best_position = tie_breaking(processing_times, e, f, ms, inserting_job, best_position, sequence_length, num_machines)
	return best_position, best_makespan


@cython.boundscheck(False)
@cython.wraparound(False)
cdef tie_breaking(int[:,:] processing_times, int[:,:] e, int[:,:] f, int[:] ms,
				  int inserting_job, int best_position, int sequence_length, int num_machines):
	"""Tie breaking when there are many insertion positions with the same makespan.

	Find the best position between n positions with the same makespan based
	on idle time approximation. Reference:
	"On insertion tie-breaking rules in heuristics for the permutation 
	flowshop scheduling problem", Computers & Operations Research 45 (2014), p60–67.

	Arguments:
		processing_times: Numpy 2d array with processing times.
		e: Completion times
		f: See reference for details
		ms: Makespan for each possible insertion position.
		inserting_job: Job to insert.
		best_position: Best insertion position based on makespan.
		sequence_length: Number of jobs in solution sequence.
		num_machines: Number of machines in this problem.

	Returns:
		best_position: Index for best insertion position.
	"""
	cdef int best_makespan, num_ties, itbp
	cdef int it, tie, i, j
	cdef int fl[801][61]

	# Save best makespan and start idle time of best position with a high value
	best_makespan = ms[best_position]
	itbp = 2000000
	num_ties = 0

	for i in range(1, sequence_length + 2):
		if ms[i] == best_makespan:
			
			it = 0
			num_ties += 1

			# If last position in sequence
			if i == sequence_length:
				for j in range(1, num_machines + 1):
					it += f[sequence_length][j] - e[sequence_length - 1][j] - processing_times[inserting_job - 1,j - 1]

			# If not last position
			else:
				fl[i][1] = f[i][1] + processing_times[i - 1][0]
				for j in range(2, num_machines + 1):
					it += f[i][j] - e[i][j] + processing_times[i - 1,j - 1] - processing_times[inserting_job - 1,j - 1]
					if fl[i][j - 1] - f[i][j] > 0:
						it += fl[i][j - 1] - f[i][j]

					if fl[i][j - 1] > f[i][j]:
						fl[i][j] = fl[i][j - 1] + processing_times[i - 1,j - 1]
					else:
						fl[i][j] = fl[i][j] + processing_times[i - 1,j - 1]

			if it < itbp:
				best_position = i
				itbp = it

	return best_position


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calculate_completion_times(int[:] sequence, int[:,:] processing_times, int num_machines, int return_array):
	"""Calculate completion times for each job in each machine.

	Arguments:
		sequence: Numpy array with Current sequence
		processing_times: Numpy 2d array with processing times.
		num_machines: Number of machines in this problem.
		return_array: If 1 return array with each completition time,
		if 0 return just an integer with the completion time of the
		last job in the last machine.

	Returns:
		e: 2d array with the completion time of each job in each machine or
		completion time of the last job in the last machine (int)
	"""
	# int variables
	cdef int sequence_length, i, j
	sequence_length = len(sequence)

	# Memory view on numpy array
	cdef int[:,::1] e = zeros((sequence_length+1,num_machines+1), dtype='int32')

	for i in range(1, sequence_length + 1):
		e[i,0] = 0
		for j in range(1, num_machines + 1):
			if i == 1:
				e[0,j] = 0

			if e[i - 1, j] > e[i, j - 1]:
				e[i, j] = e[i - 1, j] + processing_times[sequence[i - 1]-1, j-1]
			else:
				e[i, j] = e[i, j - 1] + processing_times[sequence[i - 1]-1, j-1]


	# Return completion times array or just makespan (integer)
	if return_array > 0:
		return e
	else:
		return e[sequence_length, num_machines]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calculate_idle_times(int[:] sequence, int[:,:] processing_times, int num_machines):
	"""Calculate idle times for each job in each machine.

	Arguments:
		sequence: Numpy array with Current sequence
		processing_times: Numpy 2d array with processing times.
		num_machines: Number of machines in this problem.

	Returns:
		idle_times: Assigned idle time for each job
	"""

	# int variables
	cdef int sequence_length, i, j
	sequence_length = len(sequence)

	# Memory view with numpy
	cdef int[:] idle_time = zeros((sequence_length), dtype='int32')
	cdef int[:,::1] e

	e = calculate_completion_times(sequence, processing_times, num_machines, 1)

	for i in range(0, sequence_length):
		idle_time[sequence[i] - 1] = 0
		for j in range(1, num_machines + 1):
			idle_time[sequence[i] - 1] += e[i + 1, j] - processing_times[sequence[i]-1, j-1] - e[i, j]
	return idle_time