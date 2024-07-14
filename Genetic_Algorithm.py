import random
import numpy as np
import pandas as pd
import time

def Genetic_Algorithm(problem, max_attempts=25, max_iters=100, restarts=0, pop_size=100, pop_breed_percent=0.75, elite_dreg_ratio=0.99,
                      minimum_elites=0, minimum_dregs=0, mutation_prob=0.1, record_file=None):

    def generate_pop(pop_size):
        population_list = []
        for _ in range(pop_size):
            random_state = problem.random()
            population_list.append(random_state)
        return population_list

    def find_pop_fitness(pop):
        fitness_fn = problem.fitness_fn.evaluate
        best_state = None
        best_state_fit = float('inf')
        fit_list = []

        for state in pop:
            current_fit = fitness_fn(state)
            fit_list.append(current_fit)

            if current_fit <= best_state_fit:
                best_state = state
                best_state_fit = current_fit

        return best_state_fit, fit_list, best_state

    def MateProb(pop_fit_pairs):
        remaining_pairs = [pair for pair in pop_fit_pairs]
        total_fitness = sum(-fitness for _, fitness in remaining_pairs)
        mate_probs = [(-fitness / total_fitness) for _, fitness in remaining_pairs]
        return mate_probs

    def Elites(pop_fit_pairs):
        elites = [state for state, fitness in pop_fit_pairs if fitness == best_state_fit]
        return elites

    def Dregs(pop_fit_pairs):
        num_dregs = int(len(pop) * elite_dreg_ratio)
        dregs = [state for state, fitness in pop_fit_pairs[:num_dregs]]
        return dregs

    def select_parent(pop, mate_probs):
        parent = random.choices(pop, weights=mate_probs, k=1)[0]
        return parent
    
    def rotate_right(lst, k):
        k = k % len(lst)
        return lst[-k:] + lst[:-k]

    def mutate(child, mutation_prob):
        new_child = {}

        for key in child:
            random_num = random.random()
            if random_num < mutation_prob:
                # Determine the indices i and j, and the rotation amount k
                i = random.randint(0, len(child[key]) - 1)
                j = random.randint(0, len(child[key]) - 1)

                # Ensure j is different from i
                while j == i:
                    j = random.randint(0, len(child[key]) - 1)

                k = random.randint(1, len(child[key]) - 1)  # Number of positions to rotate

                # Create the sublist
                if i < j:
                    sublist = child[key][i:j+1]
                else:
                    sublist = child[key][i:] + child[key][:j+1]

                # Rotate the sublist to the right by k positions
                rotated_sublist = rotate_right(sublist, k)

                # Apply the rotated sublist back to the child
                if i < j:
                    new_child[key] = child[key][:i] + rotated_sublist + child[key][j+1:]
                else:
                    new_child[key] = rotated_sublist[len(child[key]) - i:] + child[key][j+1:i] + rotated_sublist[:len(child[key]) - i]
            else:
                new_child[key] = child[key]

        return new_child

    def reproduce(parent_1, parent_2, mutation_prob):
        child = {}

        for key in parent_1:
            if key % 2 == 0:  # Use parent_1 for even keys
                child[key] = parent_1[key]
            else:             # Use parent_2 for odd keys
                child[key] = parent_2[key]

        # Mutate the child with the given probability
        child = mutate(child, mutation_prob)

        return child

    def genetic_algorithm(pop_size, max_attempts, max_iters):
        attempts = 0
        iters = 0
        start_time = time.time()
        iteration_data = []

        pop = generate_pop(pop_size)

        best_fit, fit_list, best_state = find_pop_fitness(pop)

        breeding_pop_size = int(pop_size * pop_breed_percent) - (minimum_elites + minimum_dregs)
        survivors_size = pop_size - breeding_pop_size
        dregs_size = max(int(survivors_size * (1.0 - elite_dreg_ratio)) if survivors_size > 1 else 0, minimum_dregs)
        elites_size = max(survivors_size - dregs_size, minimum_elites)

        if dregs_size + elites_size > survivors_size:
            over_population = dregs_size + elites_size - survivors_size
            breeding_pop_size -= over_population

        while attempts < max_attempts and iters < max_iters:
            pop_fit_pairs = list(zip(pop, fit_list))
            pop_fit_pairs.sort(key=lambda x: x[1])

            mate_probs = MateProb(pop_fit_pairs)

            next_gen = []
            for _ in range(breeding_pop_size):
                parent_1 = select_parent(pop, mate_probs)
                parent_2 = select_parent(pop, mate_probs)

                child = reproduce(parent_1, parent_2, mutation_prob)
                next_gen.append(child)

            if survivors_size > 0:
                last_gen = pop_fit_pairs
                sorted_parents = sorted(last_gen, key=lambda f: -f[1])
                best_parents = sorted_parents[:elites_size]
                next_gen.extend([p[0] for p in best_parents])
                if dregs_size > 0:
                    worst_parents = sorted_parents[-dregs_size:]
                    next_gen.extend([p[0] for p in worst_parents])

            next_gen = next_gen[:pop_size]
            pop = next_gen

            fitness_eval_start = time.time()
            current_fitness, fit_list, best_state = find_pop_fitness(pop)
            fitness_eval_end = time.time()
            fitness_time = (fitness_eval_end - fitness_eval_start)

            iters += 1
            iteration_data.append({
                'Iteration': iters,
                'Fitness': best_fit,
                'FitnessEvalTime': fitness_time,
                'Time': time.time() - start_time,
                'State': best_state
            })

            if best_fit > current_fitness:
                best_fit = current_fitness
                attempts = 0
            else:
                attempts += 1

        return best_state, best_fit, iteration_data
    
    overall_best_state = None
    overall_best_fitness = float('inf')
    overall_iteration_data = []

    for restart in range(restarts + 1):
        best_state, best_fitness, iteration_data = genetic_algorithm(pop_size, max_attempts, max_iters)

        if best_fitness < overall_best_fitness:
            overall_best_fitness = best_fitness
            overall_best_state = best_state

        iteration_data_df = pd.DataFrame(iteration_data)
        if restart == 0:
            overall_iteration_data = iteration_data_df
        else:
            overall_iteration_data = pd.concat([overall_iteration_data, iteration_data_df])

    if record_file:
        overall_iteration_data.to_csv(record_file, index=False)
 
    return overall_best_state, overall_best_fitness, overall_iteration_data

