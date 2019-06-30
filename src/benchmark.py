"""
permutation-flowshop repository

Module to import common benchmark instances used in the
literature.
"""

import numpy as np

def import_taillard():
    """Import Taillard test instances as a two dimensional numpy array.

    Instances have from 20 to 120 jobs and from 5 to 20 machines.

    Reference:
        Taillard, E.D. "Benchmarks for basic scheduling problems",
        EJOR vol. 64,pp. 78-285, 1993.
    """
    directory = "../benchmark instances/taillard instances/"
    instances_names = list()

    # Generate instance names
    for i in range(1, 121):
        if i < 10:
            index = "00" + str(i)
        elif i < 100:
            index = "0" + str(i)
        else:
            index = str(i)
        instances_names.append("ta" + index)
    return _import_files(instances_names, directory)


def import_vrf_small():
    """Import VRF small test instances as a two dimensional numpy array.

    Instances have from 10 to 60 jobs and from 5 to 20 machines.

    Reference:
        "New hard benchmark for flowshop scheduling problems minimising makespan",
        EJOR vol. 240, pp. 666-677, 2015.
    """
    directory = "../benchmark instances/vrf instances/Small/"
    instances_names = list()

    # Generate instance names
    for i in [10,20,30,40,50,60]:
        for j in [5,10,15,20]:
            for k in range(1,11):
                name = "VFR"+str(i)+"_"+str(j)+"_"+str(k)
                instances_names.append(name)
    return _import_files(instances_names, directory, vrf= True)


def import_vrf_large():
    """Import VRF large test instances as a two dimensional numpy array.

    Instances have from 100 to 800 jobs and from 20 to 60 machines.

    Reference:
        "New hard benchmark for flowshop scheduling problems minimising makespan",
        EJOR vol. 240, pp. 666-677, 2015.
    """
    directory = "../benchmark instances/vrf instances/Large/"
    instances_names = list()

    # Generate instance names
    for i in [100,200,300,400,500,600,700,800]:
        for j in [20,40,60]:
            for k in range(1, 11):
                name = "VFR" + str(i) + "_" + str(j) + "_" + str(k)
                instances_names.append(name)
    return _import_files(instances_names, directory, vrf= True)


def _import_files(instances_names, directory, vrf=False):
    """Function for loading test instances.

    Arguments:
        instances_names: File name (string).
        directory: Path to files (string).
        vrf: False for Taillard instances and True for VRF.

    Returns:
        instances: List with each instance as 2d numpy array.
    """
    instances = list()
    instance = list()

    # Read files - each file is an instance
    for _, instance_name in enumerate(instances_names):
        if vrf:
            file_name_extension = "_Gap.txt"
        else:
            file_name_extension = ""
        # Open file and jump first line
        f = open(directory + instance_name + file_name_extension, "r")
        f.readline()
        # Reset variables for each instance
        instance.clear()
        job_num = 0
        i = 0
        for line in f.readlines():
            instance.append(list())

            for item in line.split():
                if item != " " and item != "\n":
                    i += 1
                    if i % 2 == 0:
                        instance[job_num].append(int(item))
            job_num += 1
        # Create numpy 2d array for each instance and append to instances list
        instances.append(np.array(instance.copy(), dtype='int32'))
    return instances
