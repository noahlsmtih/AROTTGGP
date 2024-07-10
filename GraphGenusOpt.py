import numpy as np
import random
from mlrose_hiive.opt_probs.discrete_opt import DiscreteOpt
from GraphGenusFitness import GraphGenusFitness
import copy

class GraphGenusOpt(DiscreteOpt):
    def __init__(self, length=None, fitness_fn=None, maximize=False, max_val=2, adjacency_list=None, seed=None):
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
        
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.prob_type = 'graph_genus'

    def random(self):
        state = copy.deepcopy(self.adj_list)
        for node in state:
            lst = state[node]
            to_shuffle = lst[1:]
            shuffled_lst = [lst[0]]
            shuffled_lst.extend(np.random.permutation(to_shuffle))
            state[node] = shuffled_lst

        return state
    
    def random_neighbor(self, state):
        neighbor = copy.deepcopy(state)
        vertex = random.choice(list(neighbor.keys()))
        to_shuffle = neighbor[vertex][1:]
        shuffled_lst = [neighbor[vertex][0]]
        shuffled_lst.extend(np.random.permutation(to_shuffle))
        neighbor[vertex] = shuffled_lst

        return neighbor

    def find_neighbors(self, state):
        neighbors = []
        for node in state:
            for i in range(1, len(state[node])):
                for j in range(i + 1, len(state[node])):
                    neighbor = copy.deepcopy(state)
                    neighbor[node][i], neighbor[node][j] = neighbor[node][j], neighbor[node][i]
                    neighbors.append(neighbor)
        
        return neighbors
