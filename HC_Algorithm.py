def hill_climb(problem=None, max_iter=None):
    iter = 0
    state = problem.adjacency_list
    fitness_fn = problem.fitness_fn()
    find_neighbors = problem.find_neighbors()
    while iter<max_iter:
        neighbors=find_neighbors()
        fitness_list=[fitness_fn(neighbor) for neighbor in neighbors]
        best_neighbor_fitness=(min(fitness_list))
        if fitness_fn(state)>=best_neighbor_fitness:
            best_fit_index=fitness_list.index(min(best_neighbor_fitness))
            state=neighbors[best_fit_index]
        else:
            break
    return(fitness_fn(state), state)
