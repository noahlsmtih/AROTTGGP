import numpy as np
import mlrose_hiive as mlrose
import networkx as nx

from GraphGenusFitness import GraphGenusFitness
from GraphGenusOpt import GraphGenusOpt
from GraphGenusGenerator import GraphGenusGenerator

# Generate a new Graph Genus problem using a fixed seed.
problem = GraphGenusGenerator().generate(seed=123658, number_of_nodes=5, max_connections_per_node=4, complete_graph=True)

