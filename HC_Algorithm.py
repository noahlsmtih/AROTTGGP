import random

def hill_climb(problem=None, max_iter=None, seed=None):
    random.seed(seed)  # Set the seed for reproducibility
    iter = 0
    state = problem.adj_list
    fitness_fn = problem.fitness_fn.evaluate
    find_neighbors = problem.find_neighbors
    print("start fitness", fitness_fn(state))
    while iter < max_iter:
        neighbors = find_neighbors(state)
        fitness_list = [fitness_fn(neighbor) for neighbor in neighbors]
        print(fitness_list)
        best_neighbor_fitness = min(fitness_list)
        print(best_neighbor_fitness)
        if fitness_fn(state) >= best_neighbor_fitness:
            # Find all indices of the best fitness
            best_indices = [i for i, val in enumerate(fitness_list) if val == best_neighbor_fitness]
            # Choose a random index from the best_indices
            best_fit_index = random.choice(best_indices)
            state = neighbors[best_fit_index]
            print(iter, fitness_fn(state), state)
        else:
            break
        iter += 1
    return( fitness_fn(state), state)