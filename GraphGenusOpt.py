import numpy as np
from mlrose_hiive.opt_probs.discrete_opt import DiscreteOpt

class GraphGenusOpt(DiscreteOpt):
    def __init__(self, length, fitness_fn, maximize=False, max_val=2):
        super().__init__(length=length, fitness_fn=fitness_fn, maximize=maximize, max_val=max_val)
        self.adj_list = fitness_fn.adjacency_list  # Store adjacency list
        self.state = self.random()
        
    def reset(self):
        # Generate a random state vector
        self.state = self.random()

        # Calculate the fitness of the generated state vector
        self.fitness = self.eval_fitness(self.state)

        # Reset fitness evaluations, iteration count, and any other necessary attributes
        self.fevals = {}
        self.fitness_evaluations = 0
        self.current_iteration = 0

    def random(self):
        # Return a shuffled version of the adjacency list
        state = []
        for vertex in self.adj_list:
            state.extend(self.adj_list[vertex])  # Add all connections of the vertex
        np.random.shuffle(state)  # Shuffle the connections
        return state

    def random_neighbor(self, state):
        # Return random neighbor of the current state vector.
        # For this program, a random neighbor is a random 2-op permutation of a randomly selected vertex.
        neighbor = state.copy()
        # Select a random vertex
        vertex = np.random.randint(len(state))
        # Randomly shuffle the connections of the selected vertex
        np.random.shuffle(neighbor[vertex])
        return neighbor

    def find_neighbors(self, state):
        # Find all neighbors of the current state.
        # This should be the list of all possible 2-op permutations.
        neighbors = []
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                neighbor = state.copy()
                # Swap two connections in the vertex's edge list
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        return neighbors
