import networkx as nx
from GraphGenusGenerator import GraphGenusGenerator
from HC_Algorithm import hill_climb

seed=132456
problem = GraphGenusGenerator().generate(seed=seed, number_of_nodes=5, max_connections_per_node=3, complete_graph=True)

print(hill_climb(problem=problem, max_iter=1000, seed=seed))
