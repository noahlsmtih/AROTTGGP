import random
import time
import pandas as pd

def hill_climb(problem=None, max_iter=None, seed=None, max_attempts=25, record_file=None):
    random.seed(seed)  # Set the seed for reproducibility
    iter = 0
    state = problem.adj_list
    fitness_fn = problem.fitness_fn.evaluate
    find_neighbors = problem.find_neighbors
    attempts = 0
    
    # List to store iteration data
    iteration_data = []
    
    start_time = time.time()  # Record the start time
    
    while iter < max_iter and attempts < max_attempts:
        neighbors = find_neighbors(state)
        iteration_start_time = time.time()  # Record the start time for this iteration
        
        fitness_times = []
        fitness_list = []
        for neighbor in neighbors:
            fitness_eval_start = time.time()
            fitness = fitness_fn(neighbor)
            fitness_eval_end = time.time()
            fitness_times.append(fitness_eval_end - fitness_eval_start)
            fitness_list.append(fitness)
        
        fitness_time = sum(fitness_times) / len(fitness_times)  # Average time for fitness evaluations
        best_neighbor_fitness = min(fitness_list)
        currentFit = fitness_fn(state)
        
        if currentFit >= best_neighbor_fitness:
            if currentFit > best_neighbor_fitness:
                attempts = 0  # Reset attempts after a successful move
            else:
                attempts += 1
            # Find all indices of the best fitness
            best_indices = [i for i, val in enumerate(fitness_list) if val == best_neighbor_fitness]
            # Choose a random index from the best_indices
            best_fit_index = random.choice(best_indices)
            state = neighbors[best_fit_index]
            elapsed_time = time.time() - start_time  # Calculate total elapsed time
            iteration_data.append({
                'Iteration': iter,
                'Fitness': fitness_fn(state),
                'FitnessEvalTime': fitness_time,
                'Time': elapsed_time,
                'State': state
            })
        else:
            break
        iter += 1
    
    # Final fitness evaluation and timing
    iteration_start_time = time.time()
    final_fitness = fitness_fn(state)
    fitness_time = time.time() - iteration_start_time

    # Ensure the final iteration is included with correct fitness evaluation time
    iteration_data.append({
        'Iteration': iter,
        'Fitness': final_fitness,
        'FitnessEvalTime': fitness_time,  # Record the actual fitness evaluation time
        'Time': time.time() - start_time,  # Calculate total elapsed time
        'State': state
    })
    
    # Save iteration data to a file
    if record_file:
        df_iteration_data = pd.DataFrame(iteration_data)
        df_iteration_data.to_csv(record_file, index=False)
    
    return final_fitness, state, iteration_data
