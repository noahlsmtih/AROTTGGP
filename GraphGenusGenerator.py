import numpy as np
import networkx as nx
from GraphGenusOpt import GraphGenusOpt
from GraphGenusFitness import GraphGenusFitness

class GraphGenusGenerator:
    @staticmethod
    def create_billiard_bracket(number_of_nodes):
        if number_of_nodes < 3:
            raise ValueError("Number of nodes must be at least 3 to form a valid billiard graph.")

        # Create a bracket with relatively prime integers
        bracket = []
        current = 2
        while sum(bracket) < number_of_nodes:
            if all(np.gcd(current, b) == 1 for b in bracket):
                bracket.append(current)
            current += 1

        # Adjust the last element to ensure the sum equals the number_of_nodes
        bracket[-1] += number_of_nodes - sum(bracket)
        return bracket

    @staticmethod
    def make_billiard_graph_integers(bracket):
        B = nx.Graph()  # Empty graph
        k = len(bracket)
        n = sum(bracket)

        for i in range(2 * n):  # Add nodes to graph
            B.add_node(i)

        for rotation_vertex in range(n):  # Add edges
            ref_list = [0]
            for p in range(1, k):
                ref_list.append((ref_list[p - 1] + n - bracket[p]) % n)
            for a in ref_list:
                flip_vertex = ((a + rotation_vertex) % n) + n
                B.add_edge(rotation_vertex, flip_vertex)

        return B

    @staticmethod
    def generate(seed, number_of_nodes=20, max_connections_per_node=4, graph_type=None):
        np.random.seed(seed)
        match graph_type:
            case "complete":
                g = nx.complete_graph(number_of_nodes)
            case "complete_bipartite":
                part1_size = number_of_nodes // 2
                part2_size = number_of_nodes - part1_size
                print(f"K_({part1_size},{part2_size})")
                g = nx.complete_bipartite_graph(part1_size, part2_size)
            case "billiards":
                bracket = GraphGenusGenerator.create_billiard_bracket(number_of_nodes)
                g = GraphGenusGenerator.make_billiard_graph_integers(bracket)
            case _:
                node_connection_counts = 1 + np.random.randint(max_connections_per_node, size=number_of_nodes)
                node_connections = {}
                nodes = range(number_of_nodes)
                for n in nodes:
                    all_other_valid_nodes = [o for o in nodes if (o != n and (o not in node_connections or
                                                                            n not in node_connections[o]))]
                    count = min(node_connection_counts[n], len(all_other_valid_nodes))
                    other_nodes = sorted(np.random.choice(all_other_valid_nodes, count, replace=False))
                    node_connections[n] = [(n, o) for o in other_nodes]

                g = nx.Graph()
                g.add_edges_from([x for y in node_connections.values() for x in y])

                for n in nodes:
                    cannot_reach = [(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()]
                    for s, f in cannot_reach:
                        g.add_edge(s, f)
                        check_reach = len([(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()])
                        if check_reach == 0:
                            break

        adjacency_list = {node: list(g.neighbors(node)) for node in g.nodes()}

        adj_dict = {}
        for node in g.nodes():
            adj_list = list(g.neighbors(node))
            adj_dict[node] = adj_list

        problem = GraphGenusOpt(length=number_of_nodes, fitness_fn=GraphGenusFitness(adjacency_list=adj_dict), maximize=False, adjacency_list=adjacency_list, seed=seed)
        return problem
