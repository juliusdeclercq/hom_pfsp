# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:31:05 2024

@author: Julius de Clercq
"""

import os
import argparse
import pandas as pd
import numpy as np
from time import time as t
from concurrent.futures import ProcessPoolExecutor

#%%         Read the instances

def read_instances(filename):
    """
    Function to read the instances from the Excel file. The processing p
    are written to sheets that are named after the corresponding instance file.
    The Excel file is written with the instance_extractor.py script.
    """
    instances_dict = {}
    excel_file = pd.ExcelFile(filename)

    for sheet_name in excel_file.sheet_names:
        if sheet_name == "Instances": # Skipping the first sheet containing the summary of the instances.
            continue

        df = pd.read_excel(filename, sheet_name=sheet_name, index_col=0)

        if sheet_name not in instances_dict:
            instances_dict[sheet_name] = np.array(df)
    return instances_dict


#%%

def update_z(new_sequence):
    """
    Update z based on the new job sequence, which is a list.
    """
    N = len(new_sequence)
    new_z = np.zeros((N, N))
    for j in range(N):
        new_z[new_sequence[j], j] = 1
    return new_z

#%%

def get_makespan(p, z_list):
    """
    Calculate the makespan for a given schedule and instance.
    """
    jobs = sorted(z_list)
    N, M = p[jobs].shape
    s = np.zeros((N, M))

    # Fill the completion p matrix
    for j in range(N):
        for m in range(M):
            if j == 0 and m == 0:
                s[j, m] = p[z_list[j], m]
            elif j == 0:
                s[j, m] = s[j, m-1] + p[z_list[j], m]
            elif m == 0:
                s[j, m] = s[j-1, m] + p[z_list[j], m]
            else:
                s[j, m] = max(s[j-1, m], s[j, m-1]) + p[z_list[j], m]

    return s, s[-1, -1]



#%%
def neh_heuristic(p):
    """
    Heuristic according to Nawal et al. (1982), using the acceleration from
    Taillard (1990).
    Acceleration largely based on https://github.com/mattianeroni/NEH/blob/main/src/solver.py.
    """
    N, M = p.shape

    # Step 1: Calculate total processing p for each job
    # Step 2: Sort jobs in descending order of total processing time
    job_indices = list(range(N))
    sorted_jobs = [(i, p[i].sum()) for i in job_indices]
    sorted_jobs.sort(key=lambda x: x[1], reverse=True)  # Sort by total processing time

    # Step 3: Start with only the first job. In the first iteration, only the first
    # two jobs in the sorted list are considered.
    sequence = [sorted_jobs[0][0]]

    # Step 4 with Taillard's acceleration.
    for k, job in zip(range(1, N), [job[0] for job in sorted_jobs[1:]]):
            for i in range(k+1):
                candidate_sequence = sequence.copy()
                candidate_sequence.insert(i, job)

            # Init earliest completion time of i-th job on j-th machine
            e = np.zeros((k+2, M+1))
            # Init the tail of the i-th job on the j-th machine
            q = np.zeros((k+2, M+1))
            # Init the earliest relative completion time for the k-th job
            # in i-th position on j-th machine.
            f = np.zeros((k+2, M+1))
            # For each position in which the job can be inserted...
            for i in range(k + 1):
                # Compute the earliest completion time, the tail, and the relative completion time
                for j in range(M):
                    if i < k:
                        e[i, j] = max(e[i, j-1], e[i-1, j]) + p[sequence[i], j]
                    if i > 0:
                        q[k-i, M-j-1] = max(q[k-i, M-j], q[k-i+1, M-j-1]) + p[sequence[k-i], M-j-1]
                    f[i, j] = max(f[i, j-1], e[i-1, j]) + p[job, j]
                    # except Exception as e:
                    #     print(f"i = {i}\nj = {j} \njob = {job}")
                    #     raise e

            # Partial makespans inserting job in i-th position
            candidate_makespan = np.amax(f + q, axis=1)[:-1]
            # Find the position where to insert k-th job that minimise the makespan
            position = np.where(candidate_makespan == candidate_makespan.min())[0][0]
            makespan = int(candidate_makespan[position])
            # Insert the k-th job in the position that minimised the partial makespan
            sequence.insert(position, job)
            # print("Selected: ", candidate_makespan,  sequence, makespan, get_makespan(p, sequence))
            # print("--------------------")
            #if k > 2: break

    # print(sorted_jobs)
    # print(sequence, makespan)
    return sequence, int(makespan)

#%%

def insertion_LS(rng, p, z, makespan, verbose = False):
    """
    Best-improvement insertion local search operator.
    """
    N, M = p.shape

    improvements = 0        # Nice to keep track of how many improvements were

    best_sequence = z.copy()
    best_makespan = makespan

    sequence = z.copy()
    insertion_list = z.copy()
    # Shuffle the list to randomize which job to insert.
    rng.shuffle(insertion_list)

    improved = True
    iteration = 0
    while improved: # The algorithm will stop when no job could be inserted in a better place.
        improved = False
        if verbose:
            iteration += 1
            print(f"Iteration = {iteration}")
        for i in insertion_list:
            insertion_dict = {}
            current_idx = sequence.index(i)
            rm_sequence = sequence.copy()
            rm_sequence.remove(i)
            for j in range(N):
                if j == current_idx:    # Skipping the current position
                    continue
                new_sequence = rm_sequence[:j] + [i] + rm_sequence[j:]
                _, makespan = get_makespan(p, new_sequence)
                insertion_dict.update({j : makespan})

            best_insertion = min(insertion_dict, key=insertion_dict.get)
            min_makespan = insertion_dict[best_insertion]
            if min_makespan < best_makespan:
                improved = True
                if verbose:
                    print("Improved!")
                improvements += 1

                best_makespan = min_makespan
                best_sequence = rm_sequence[:j] + [i] + rm_sequence[j:]
                sequence = best_sequence.copy()

    return best_sequence, int(best_makespan)


#%%

def destruction_construction_LS(rng, p, z, makespan, d = 4, pre_insertion = False, verbose = False):
    """
    Destruction-construction Local Search operator.
    """
    N, M = p.shape

    # Find the items to remove. The set of items is D, and |D| = d.
    D = z.copy()
    rng.shuffle(D)
    D = D[:d]

    # Remove the items from the sequence
    rm_sequence = [job for job in z if job not in D]
    _, makespan = get_makespan(p, rm_sequence)

    if pre_insertion: # Optional optimization of the destroyed sequence before reconstruction.
        rm_sequence, makespan = insertion_LS(rng, p, rm_sequence, makespan)

    strt = t()
    for i in D: # Inserting the removed jobs, which are already in random order.
        insertion_dict = {}
        for j in range(len(rm_sequence) + 1):
            new_sequence = rm_sequence[:j] + [i] + rm_sequence[j:]
            _, makespan = get_makespan(p, new_sequence)
            insertion_dict.update({j : makespan})

        # Insert the removed job in the best position.
        best_insertion = min(insertion_dict, key=insertion_dict.get)
        rm_sequence = rm_sequence[:best_insertion] + [i] + rm_sequence[best_insertion:]


    if verbose:
        reconstruction_time = t() - strt
        print(f"Reconstruction took {round(reconstruction_time)} seconds.")
    # Find the makespan after reconstruction is complete.
    _, makespan = get_makespan(p, rm_sequence)

    return rm_sequence, int(makespan)


#%%

def swap_LS(rng, p, z, makespan, verbose = False):
    """
    Swap, or 2-OPT, Local Search operator. Randomly selects a job and swaps it
    with another if it improves the makespan best of all possible swaps for this
    job.
    Much alike to the Insertion LS Operator.

    Also implementing a Tabu list to avoid redundant computation. On this Tabu
    list, the starting sequence and the accepted sequences are kept. This to
    avoid extreme memory usage while still avoiding cycles in the LS.
    """
    N, M = p.shape

    sequence = z.copy()
    best_makespan = makespan

    # Swap the jobs in random order.
    swap_sequence = z.copy()
    rng.shuffle(swap_sequence)

    # Tabu list for avoiding redundant computation.
    # Keeping sequences we started with or had accepted before.
    tabu_list = [z]

    improved = True
    iteration = 0
    while improved:
        improved = False

        if verbose:
            iteration += 1
            print(f"Iteration = {iteration}")
        for i in swap_sequence:
            swap_dict = {} # Dictionary to keep track of the makespans.
            i_idx = sequence.index(i)
            for j in sequence:
                if i == j: # We do not swap a job with itself.
                    continue
                # Make the swap.
                j_idx = sequence.index(j)
                candidate_sequence = sequence.copy()
                candidate_sequence[i_idx], candidate_sequence[j_idx] = j, i
                if candidate_sequence in tabu_list: # Skipping sequences we already had.
                    continue

                _, makespan = get_makespan(p, candidate_sequence)
                swap_dict.update({j : makespan})

            best_swap = min(swap_dict, key=swap_dict.get)
            min_makespan = swap_dict[best_swap]
            if min_makespan < best_makespan:
                improved = True
                if verbose:
                    print("Improved!")
                best_makespan = min_makespan
                swap_idx = sequence.index(best_swap)
                sequence[i_idx], sequence[swap_idx] = best_swap, i
                tabu_list.append(sequence)

    return z, int(best_makespan)


#%%

def single_ILS(params):
    """
    Implementation of the ILS with only a single operator. As the perturbation operator
    and the local search operators are all operators, and I can only choose a single
    one, making a logical choice is not so straightforward. I choose the Swap
    operator, due to its LS nature and randomization. Due to the randomization it
    can still escape local minima and due to its LS nature it can still find minima.
    """
    strt = t()
    p = params[0]
    rng, T, d, iter_limit, verbose = params[1]
    N, M = p.shape
    Const_Temperature = T * p.sum() / (N * M * 10)

    # Generate initial solution
    z, makespan = neh_heuristic(p)
    # sequence, makespan = swap_LS(rng, p, z, makespan, verbose = False)
    sequence, makespan = destruction_construction_LS(rng, p, z, makespan, d = d, pre_insertion = True, verbose = False)

    termination_condition = False
    convergence = [makespan]
    iteration = 0
    while not termination_condition:
        non_improvements = 0
        iteration += 1
        if verbose:
            print(f"Iteration = {iteration}")
            if iteration % 5 == 0:
                run_time = t() - strt
                print(f"Current run time: {int(run_time / 60)} minutes and {round(run_time % 60)} seconds.\n")

        new_sequence, new_makespan = destruction_construction_LS(rng, p, sequence, makespan, d = d, pre_insertion = True, verbose = False)

        accept_probability = np.exp((makespan - new_makespan)/Const_Temperature)

        if rng.random() < accept_probability:
            sequence, makespan = new_sequence.copy(), new_makespan
        else:
            non_improvements += 1
            if (verbose and non_improvements % 2 == 0):
                print(f"Now at {non_improvements} non-improvements.\n")

        if (iteration == iter_limit or non_improvements == 5):
            termination_condition = True

        convergence.append(makespan)

    run_time = t() - strt
    results = {"sequence": sequence,
               "makespan": makespan,
               "run_time": run_time,
               "convergence": convergence}
    return results

#%%

def ILS(params):
    """
    Iterated Local Search (ILS) metaheuristic. Using constant temperature as in
    the Iterated Greedy algorithm of Ruiz & Stültze (2007) and the ILS implemenation
    from Karimi-Mamaghan (2022). Standard parameter values are also taken from
    Ruiz & Stültze (2007).

    Parameters
    ----------
    p : np.array
        Processing times matrix
    T : float
        Temperature scale. The default is 0.4.
    d : int
        Destruction size. The default is 4.

    """
    strt = t()
    # Unpack parameters
    p = params[0]
    rng, T, d, iter_limit, verbose = params[1]
    N, M = p.shape
    Const_Temperature = T * p.sum() / (N * M * 10)


    # Generate initial solution
    z, makespan = neh_heuristic(p)

    # Initial LS: Swap
    sequence, makespan = swap_LS(rng, p, z, makespan, verbose = False)

    termination_condition = False
    convergence = [makespan]
    iteration = 0
    while not termination_condition:
        non_improvements = 0
        iteration += 1
        if verbose:
            print(f"Iteration = {iteration}")
            if iteration % 5 == 0:
                run_time = t() - strt
                print(f"Current run time: {int(run_time / 60)} minutes and {round(run_time % 60)} seconds.\n")
        # Perturbation: Destruction-Construction
        new_sequence, new_makespan = destruction_construction_LS(rng, p, sequence, makespan, d = d, pre_insertion = False, verbose = False)

        # Local Search: Insertion
        new_sequence, new_makespan = insertion_LS(rng, p, new_sequence, new_makespan, verbose = False)

        # Metropolis Acceptance probability
        accept_probability = np.exp((makespan - new_makespan)/Const_Temperature)
        # print(accept_probability)

        if rng.random() < accept_probability:
            # print("Acceptance!")
            sequence, makespan = new_sequence.copy(), new_makespan
        else:
            non_improvements += 1
            if (verbose and non_improvements % 2 == 0):
                print(f"Now at {non_improvements} non-improvements.\n")

        if iteration == iter_limit:
            termination_condition = True
        convergence.append(makespan)

    run_time = t() - strt
    results = {"sequence": sequence,
               "makespan": makespan,
               "run_time": run_time,
               "convergence": convergence}
    return results





#%%


def Q_learning(rng, C_prev, C_prev_best, makespan, best_makespan, epsilon, beta, alpha, gamma, eta, A, s, d, Q):
    """
    Q-learning function taken from the pseudo code of Algorithm 3 from
    Karimi-Mamaghan et al. (2023).
    """
    # Reward function (equations 10-12 from K-M e.a. (2023))
    r_local = max(C_prev - makespan, 0)/C_prev
    r_global = max(C_prev_best - best_makespan, 0)/C_prev_best
    reward = eta * r_local + (1 - eta) * r_global

    # Did we reach a local optimum?
    if best_makespan < C_prev_best:
        s_new = 1
    else:
        s_new = 0

    # Update Q-table!
    Q[s, A.index(d)] = Q[s, A.index(d)] + alpha * (reward + gamma * Q[s_new].max() - Q[s, A.index(d)])

    epsilon = epsilon * beta

    if rng.random() > epsilon: # Usually we exploit
        d_new = A[Q[s_new].argmax()]
    else: # Sometimes we select a random action from A
        random_action = [i for i in range(len(A))]
        rng.shuffle(random_action)
        d_new = A[random_action[0]]

    return Q, s, d_new


#%%

def QILS(params):
    """
    Q-learning implementation of the ILS algorithm, following Karimi-Mamaghan (2022)
    but mostly Karimi-Mamaghan et al. (2023). As in the latter, states are a binary,
    defining whether the search has gotten stuck in a local minimum. Default
    parameter values are taken as the tuned values from KM et al. 2023.

    Parameters
    ----------
    p : np.array
        Processing times.
    A : list
        Action set.
    epsilon : TYPE, optional
        Epsilon-greedy exploration probability. The default is 0.8.
    beta : TYPE, optional
        Epsilon decay rate. The default is 0.996.
    alpha : TYPE, optional
        Learning rate. The default is 0.6.
    gamma : TYPE, optional
        Discount factor. The default is 0.8.
    E : TYPE, optional
        Episode size. The default is 6.
    eta : TYPE, optional
        Local/global improvement weight. The default is 0.3.
    T : TYPE, optional
        Temperature scale. The default is 0.4.
    iter_limit : TYPE, optional
        DESCRIPTION. The default is 100.
    verbose : bool, optional
        Whether or not to execute print statements. The default is False.

    Returns
    -------
    None.

    """
    strt = t()
    # Unpack parameters
    p = params[0]
    rng, A, epsilon, beta, alpha, gamma, E, eta, T, iter_limit, verbose = params[1]

    N, M = p.shape
    Const_Temperature = T * p.sum() / (N * M * 10)
    S = [0,1]

    # Generate initial solution
    z, makespan = neh_heuristic(p)

    # Initial LS: Swap
    sequence, makespan = swap_LS(rng, p, z, makespan, verbose = False)
    best_sequence = sequence.copy()
    convergence = [makespan]

    # Initialize Q-matrix with zeroes
    Q = np.zeros((len(S), len(A)))
    s = 0   # Set initial state to zero (1 means local minimum reached)

    # Select a random action a from A
    random_action = [i for i in range(len(A))]
    rng.shuffle(random_action)
    d = A[random_action[0]]

    termination_condition = False
    episode = 0
    iteration = 0
    while not termination_condition:
        # Update makespans
        C_prev = makespan
        _, C_prev_best = get_makespan(p, best_sequence)
        best_makespan = C_prev_best

        # Start of an episode.
        episode += 1
        if verbose:
            print(f"Episode {episode}")
        for i in range(E):
            iteration += 1
            if (verbose and iteration % 5 == 0):
                run_time = t() - strt
                print(f"Current run time: {int(run_time / 60)} minutes and {round(run_time % 60)} seconds.\n")

            # Destruction-Construction perturbation
            new_sequence, new_makespan = destruction_construction_LS(rng, p, sequence, makespan, d = d, pre_insertion = False, verbose = False)

            # Local Search: Insertion
            new_sequence, new_makespan = insertion_LS(rng, p, new_sequence, new_makespan, verbose = False)

            # Metropolis Acceptance probability
            accept_probability = np.exp((makespan - new_makespan)/Const_Temperature)
            # print(accept_probability)

            if rng.random() < accept_probability:
                if verbose:
                    print("Acceptance!")
                sequence = new_sequence.copy()
                makespan = min(makespan, new_makespan)  # Tracking the best local optimum of this episode.

            if new_makespan < best_makespan:
                best_sequence = new_sequence.copy()
                best_makespan = new_makespan

            convergence.append(best_makespan)

        Q, s, d = Q_learning(rng, C_prev, C_prev_best, makespan, best_makespan, epsilon, beta, alpha, gamma, eta, A, s, d, Q)

        if iteration == iter_limit:
            termination_condition = True

    run_time = t() - strt
    results = {"sequence": best_sequence,
               "makespan": best_makespan,
               "run_time": run_time,
               "convergence": convergence}
    return results



#%%

def metaheuristic_random_parallel_worker(parameters):
    """
    Need to define a worker function to make sure that the processes do not all
    get the same instance of the random number generator. Using numpy's seed
    sequence which hopefully saves reproducibility.
    """
    metaheuristic = parameters[0]
    parameters = parameters[1]
    seed_seq = np.random.SeedSequence()
    rng = np.random.default_rng(seed_seq.generate_state(1)[0])
    parameters[1] = [rng] + parameters[1]

    return metaheuristic(parameters)


def process_instances(instances, metaheur, params, instance_runs, verbose = False):
    output = []
    metaheuristic = globals()[metaheur]
    # ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers = os.cpu_count() - 1)  as executor: # Use as many workers as there are CPU cores - 1.
        futures = {
            executor.submit(metaheuristic_random_parallel_worker, [metaheuristic, [p, params]]): instance
            for instance, p in instances.items() for i in range(instance_runs)
        }

        for future in futures:
            instance = futures[future]
            results = future.result()
            print(instance)
            output.append({
                'instance': instance,
                'sequence' : str(results["sequence"]),
                'makespan'  : results["makespan"],
                'run_time' : results["run_time"],
                'convergence' : str(results["convergence"]),
                })

    results_df = pd.DataFrame(output)
    return results_df


#%%             Argument parser
def parse_arguments():
    """
    This is to handle the input_dir and output_dir arguments that should be passed
    to the script when executing it from the command line or a bash file. This
    is necessary for execution from Snellius. To execute on your own (Windows)
    machine, go to cmd, set the working directory to the directory of the script,
    and run on the command line:
        python HOM_final_assignment.py "%cd%" "%cd%"
    Otherwise try:
        py HOM_final_assignment.py "%cd%" "%cd%"
    Make sure Instance.xlsx and Instance_test.xlsx are situated in the same
    directory as the script.
    """
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process input file and save results to output directory.")

    # Add arguments
    parser.add_argument('input_dir',  nargs='?', type=str, help="Path to the input directory")
    parser.add_argument('output_dir', nargs='?', type=str, help="Path to the output directory")

    # Parse arguments
    try:
        args = parser.parse_args()
        input_dir = args.input_dir
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except Exception as e:
        print("\n\nInput and output directory arguments not passed to script.\n\n")
        print(e)
        input_dir = output_dir = ""
        print("\nThe script is flexible enough that this is no issue.\n")

    return input_dir, output_dir

#%%             Main
###########################################################
### main
def main():
    global_start = t()

    ###############
    # Magic numbers

    # Limits
    iter_limit = 600
    instance_runs = 5

    # Parameter values
    T = 0.4
    d = 4
    A = [1, 2, 5]
    epsilon = 0.8
    beta = 0.996
    alpha = 0.6
    gamma = 0.8
    E = 6
    eta = 0.3



    # Read the input and output directory arguments given to the script.
    input_dir, output_dir = parse_arguments()

    # Set test and new to false for the VRF-hard-large instance set.
    test = True
    new = False
    if test:
        filename = "Instance_test.xlsx"
        # instance = "VRF60_20_1"
    elif new:
        filename = "Instance_new.xlsx"
    else:
        filename = "Instance.xlsx"  # This is the one with the VRF-hard-large instances.
        # instance = "VRF100_20_10"

    # Read instances
    input_file = os.path.join(input_dir, filename)

    try:
        instances = read_instances(input_file)
    except FileNotFoundError:
        print(f"Input file {input_file} not found.")
        return

    instances = read_instances(filename)
    # p = instances[instance]
    # instances = {key: value for i, (key, value) in enumerate(instances.items()) if i < 3}

    parameters = {"single_ILS" : [T, d, iter_limit, False],
                  "ILS": [T, d, iter_limit, False],
                  "QILS": [A, epsilon, beta, alpha, gamma, E, eta, T, iter_limit, True]}

    # startrun = t()
    # z, makespan = neh_heuristic(p)
    # print(f"Initial makespan = {makespan}")
    # # results = swap_ILS((p, parameters['swap_ILS']))
    # # results = ILS((p, parameters['ILS']))
    # results = QILS((p, parameters['QILS']))
    # run_time = t()
    # print(f"Final makespan = {results['makespan']}")

    for metaheur, params in parameters.items():
        results_df = process_instances(instances, metaheur, params, instance_runs)
        output_file = os.path.join(output_dir, f'Results_{metaheur}.xlsx')
        results_df.to_excel(output_file, index=False)
        print(f"\n\nMetaheuristic {metaheur} is finished!!!\n\n")

    global_time = t() - global_start
    print(f"\n\n\n\n\n{global_time}\n\n\n\n\n")



#%%             Start main
###########################################################
### start main
if __name__ == '__main__':
    main()






