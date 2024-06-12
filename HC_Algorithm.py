import random
import time
import pandas as pd

def hill_climb(problem=None, max_iter=None, seed=None, max_attempts = 25, record_file=None):
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
        fitness_list = [fitness_fn(neighbor) for neighbor in neighbors]
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
            iteration_data.append({
                'Iteration': iter,
                'Fitness': fitness_fn(state),
                'Time': time.time() - start_time,  # Calculate time difference
                'State': state
            })
        else:
            break
        iter += 1
    
    # Ensure the final iteration is included
    iteration_data.append({
        'Iteration': iter,
        'Fitness': fitness_fn(state),
        'Time': time.time() - start_time,  # Calculate time difference
        'State': state
    })
    
    # Save iteration data to a file
    if record_file:
        df_iteration_data = pd.DataFrame(iteration_data)
        df_iteration_data.to_csv(record_file, index=False)
    
    return fitness_fn(state), state, iteration_data