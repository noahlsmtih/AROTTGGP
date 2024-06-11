import numpy as np
from mlrose_hiive.opt_probs.discrete_opt import DiscreteOpt
from GraphGenusFitness import GraphGenusFitness

class GraphGenusOpt(DiscreteOpt):
    def __init__(self, length=None, fitness_fn=None, maximize=False, max_val=2, adjacency_list=None):
        if (fitness_fn is None) and (adjacency_list is None):
            raise Exception("At least one of fitness_fn and adjacency_list must be specified.")
        elif fitness_fn is None:
            fitness_fn = GraphGenusFitness(adjacency_list)
        self.adj_list = adjacency_list
        if length is None:
            if adjacency_list is not None:
                length = len(adjacency_list)
        self.length = length
        super().__init__(length=length, fitness_fn=fitness_fn, maximize=maximize, max_val=max_val)
        
        if self.fitness_fn.get_prob_type() != 'discrete':
            raise Exception("fitness_fn must have problem type 'discrete'.")
            
        self.prob_type = 'graph_genus'

    def reset(self):
        self.state = self.random()
        self.fitness = self.eval_fitness(self.state)
        self.fevals = {}
        self.fitness_evaluations = 0
        self.current_iteration = 0

    def random(self):
        return np.random.randint(0, 2, self.length)

    def random_neighbor(self, state):
        neighbor = state.copy()
        vertex = np.random.randint(len(state))
        neighbor[vertex] = 1 - neighbor[vertex]  # Flip the bit
        return neighbor

    def find_neighbors(self):
        neighbors = []
        for i in range(len(self)):
            neighbor = self.copy()
            neighbor[i] = 1 - neighbor[i]  # Flip the bit
            neighbors.append(neighbor)
        return neighbors
