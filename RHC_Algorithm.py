import random
import time
import pandas as pd
import numpy as np

def is_power_of_two(n):
    return (n & (n - 1) == 0) and n != 0

def get_best_result(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Sort the DataFrame by fitness and time
    df_sorted = df.sort_values(by=['Fitness', 'Time'])

    # Get the result with the lowest time while still having the lowest fitness
    best_result = df_sorted.iloc[0]

    return best_result

def random_hill_climb(problem, max_attempts=25, max_iters=0, restarts=0, init_state=None, curve=False, random_state=None, record_file=None):
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    fitness_fn = problem.fitness_fn.evaluate
    find_neighbors = problem.find_neighbors

    def hill_climb_single_run(initial_state, max_attempts, max_iters):
        state = initial_state
        best_state = state
        best_fitness = fitness_fn(state)
        iter = 0
        attempts = 0
        iteration_data = []
        start_time = time.time()
        
        while iter < max_iters and attempts < max_attempts:
            neighbors = find_neighbors(state)
            if not neighbors:
                break

            fitness_times = []
            fitness_list = []
            for neighbor in neighbors:
                fitness_eval_start = time.time()
                fitness = fitness_fn(neighbor)
                fitness_eval_end = time.time()
                fitness_times.append(fitness_eval_end - fitness_eval_start)
                fitness_list.append(fitness)

            avg_fitness_time = sum(fitness_times) / len(fitness_times) if fitness_times else 0
            best_neighbor_fitness = min(fitness_list)
            
            if best_neighbor_fitness < best_fitness:
                best_indices = [i for i, val in enumerate(fitness_list) if val == best_neighbor_fitness]
                best_fit_index = random.choice(best_indices)
                state = neighbors[best_fit_index]
                best_fitness = best_neighbor_fitness
                best_state = state
                attempts = 0  # Reset attempts after a successful move
            else:
                attempts += 1

            iter += 1
            iteration_data.append({
                'Iteration': iter,
                'Fitness': best_fitness,
                'FitnessEvalTime': avg_fitness_time,
                'Time': time.time() - start_time,
                'State': state
            })

        return best_state, best_fitness, iteration_data

    overall_best_state = None
    overall_best_fitness = float('inf')
    overall_iteration_data = []

    for restart in range(restarts + 1):
        state = problem.random()
        print("restarted")
        best_state, best_fitness, iteration_data = hill_climb_single_run(state, max_attempts, max_iters)

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
 
    if curve:
        fitness_curve = [data['Fitness'] for data in overall_iteration_data]
        return overall_best_state, overall_best_fitness, fitness_curve, overall_iteration_data

    return overall_best_state, overall_best_fitness, overall_iteration_data

# Example usage:
# problem = YourProblemClass()  # Ensure your problem class has the required methods
# best_state, best_fitness, iteration_data = random_hill_climb(problem, max_iters=1000, restarts=10, random_state=42, record_file="random_hill_climb_iterations.csv")
