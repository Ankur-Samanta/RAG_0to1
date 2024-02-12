import numpy as np
from scipy.spatial import distance

class Node:
    def __init__(self, data, id, max_layer):
        self.data = data
        self.id = id
        self.max_layer = max_layer  # Correctly include max_layer
        self.neighbors = {i: [] for i in range(max_layer + 1)}


def euclidean_distance(a, b):
    return distance.euclidean(a.data, b.data)

class HNSW:
    def __init__(self, max_layer, M):
        self.nodes = []
        self.entry_point = None
        self.max_layer = max_layer
        self.M = M  # Max neighbors per node per layer

    def _insert_node(self, new_node, layer):
        if not self.nodes:  # First node becomes the entry point
            self.nodes.append(new_node)
            self.entry_point = new_node
            return

        # Search for nearest neighbors in the graph starting from the entry point
        current_node = self.entry_point
        for l in range(self.max_layer, -1, -1):
            # Find the closest node at the current layer
            closest_node, _ = min(((node, euclidean_distance(new_node, node)) for node in self.nodes if l in node.neighbors), key=lambda x: x[1])
            current_node = closest_node

        # Update neighbors for the new node across layers
        for l in range(layer + 1):
            neighbors = self._get_neighbors(new_node, l)
            new_node.neighbors[l] = neighbors
            for neighbor in neighbors:
                if len(neighbor.neighbors[l]) < self.M:
                    neighbor.neighbors[l].append(new_node)
                else:
                    # If neighbor has M neighbors, replace the farthest if the new node is closer
                    farthest_neighbor = max(neighbor.neighbors[l], key=lambda x: euclidean_distance(x, neighbor))
                    if euclidean_distance(new_node, neighbor) < euclidean_distance(farthest_neighbor, neighbor):
                        neighbor.neighbors[l].remove(farthest_neighbor)
                        neighbor.neighbors[l].append(new_node)

    def _get_neighbors(self, node, layer):
        """Find up to M nearest neighbors of 'node' at 'layer'."""
        all_nodes = [n for n in self.nodes if layer in n.neighbors and n.id != node.id]
        if not all_nodes:
            return []
        distances = [(n, euclidean_distance(node, n)) for n in all_nodes]
        distances.sort(key=lambda x: x[1])
        neighbors = [n for n, _ in distances[:self.M]]
        return neighbors

    def add_point(self, data, id):
        node_layer = np.random.randint(self.max_layer + 1)  # Random layer assignment
        new_node = Node(data, id, node_layer)
        self._insert_node(new_node, node_layer)
        self.nodes.append(new_node)
        # Update the entry point if the new node's layer is higher
        if self.entry_point is None or node_layer > self.entry_point.max_layer:
            self.entry_point = new_node

    def search_knn(self, query_data, k, ef=10):
        """Search for k nearest neighbors of 'query_data' using ef as the size of the dynamic candidate list."""
        if not self.nodes:
            return []

        query_node = Node(query_data, -1, 0)
        current_node = self.entry_point
        for l in range(self.max_layer, -1, -1):
            # Find the closest node at the current layer
            while True:
                closer_node = min(current_node.neighbors[l], key=lambda x: euclidean_distance(query_node, x), default=None)
                if closer_node and euclidean_distance(query_node, closer_node) < euclidean_distance(query_node, current_node):
                    current_node = closer_node
                else:
                    break
        
        # Conducting the search in the 0 layer
        candidates = set(current_node.neighbors[0])
        candidates_distance = [(node, euclidean_distance(query_node, node)) for node in candidates]
        candidates_distance.sort(key=lambda x: x[1])

        # Return the top k closest nodes
        return [node.id for node, _ in candidates_distance[:k]]

# Example usage
np.random.seed(42)
embeddings = np.random.rand(100, 5)  # 100 5-dimensional embeddings
ann = HNSW(max_layer=5, M=5)
for i, emb in enumerate(embeddings):
    ann.add_point(emb, i)

query_embedding = np.random.rand(5)
k = 5
nearest_indices = ann.search_knn(query_embedding, k)

print(f"Nearest indices: {nearest_indices}")
