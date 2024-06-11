class GraphGenusFitness:
    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list

    def evaluate(self, state):
        def findOrderedPairs(adj_list):
            pairs = [(i,j) for i in range(len(adj_list)) for j in adj_list[i]]
            return pairs

        def countFaces(ord_pairs, adj_list):
            faces = []
            while ord_pairs:
                i = ord_pairs.pop(0)
                face = [i]
                next_edge = None
                while next_edge != face[0]:
                    index = adj_list[i[1]].index(i[0])
                    next_vertex = adj_list[i[1]][(index + 1) % len(adj_list[i[1]])]
                    next_edge = (i[1], next_vertex)
                    if next_edge != face[0]:
                        face.append(next_edge)
                        if next_edge in ord_pairs:
                            ord_pairs.remove(next_edge)
                        i = next_edge
                faces.append(face)
            return len(faces)

        orderedPairs = findOrderedPairs(state)

        V = len(state)
        E = sum([len(state[i]) for i in range(V)]) / 2
        F = countFaces(orderedPairs, state)

        genus = int((2 - (V - E + F)) / 2)
        return genus

    def get_prob_type(self):
        return 'discrete'
