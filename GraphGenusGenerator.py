import numpy as np
import networkx as nx
from GraphGenusOpt import GraphGenusOpt
from GraphGenusFitness import GraphGenusFitness

class GraphGenusGenerator:
    @staticmethod
    def generate(seed, number_of_nodes=20, max_connections_per_node=4, maximize=False):

        np.random.seed(seed)
        # All nodes have to be connected, somehow.
        node_connection_counts = 1 + np.random.randint(max_connections_per_node, size=number_of_nodes)

        node_connections = {}
        nodes = range(number_of_nodes)
        for n in nodes:
            all_other_valid_nodes = [o for o in nodes if (o != n and (o not in node_connections or
                                                                      n not in node_connections[o]))]
            count = min(node_connection_counts[n], len(all_other_valid_nodes))
            other_nodes = sorted(np.random.choice(all_other_valid_nodes, count, replace=False))
            node_connections[n] = [(n, o) for o in other_nodes]

        # Check connectivity
        g = nx.Graph()
        g.add_edges_from([x for y in node_connections.values() for x in y])

        for n in nodes:
            cannot_reach = [(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()]
            for s, f in cannot_reach:
                g.add_edge(s, f)
                check_reach = len([(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()])
                if check_reach == 0:
                    break

        edges = [(s, f) for (s, f) in g.edges()]
        problem = DiscreteOpt(length=len(edges), fitness_fn=GraphGenusFitness(adjacency_list=g), maximize=maximize)
        return problem