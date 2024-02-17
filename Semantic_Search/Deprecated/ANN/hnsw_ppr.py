import numpy as np
from scipy.spatial import distance
import os
import pickle

class Node:
    def __init__(self, data, id, max_layer, doc_id=None):
        self.data = data
        self.id = id
        self.max_layer = max_layer
        self.doc_id = doc_id
        self.neighbors = {i: [] for i in range(max_layer + 1)}

class HNSW:
    def __init__(self, initial_dataset_size, M=16, Mmax=None):
        self.nodes = []
        self.entry_point = None
        self.dataset_size = initial_dataset_size
        self.max_layer = int(np.log2(initial_dataset_size))
        self.M = M
        self.Mmax = Mmax if Mmax is not None else M * 2  # Default to 2*M if not specified

    def calculate_max_layer(self, dataset_size):
        return int(np.log2(dataset_size))

    def euclidean_distance(self, a, b):
        return distance.euclidean(a, b)

    def _get_layer(self):
        l = 0
        while np.random.rand() < np.exp(-l) and l < self.max_layer:
            l += 1
        return l

    def _insert_node(self, new_node):
        if not self.nodes:
            self.nodes.append(new_node)
            self.entry_point = new_node
            return

        entry_point = self.entry_point
        for l in reversed(range(new_node.max_layer + 1)):
            found_neighbors = self.search_layer(new_node.data, 1 if l > 0 else self.M, l)
            new_node.neighbors[l] = found_neighbors[:self.M]
            for neighbor in found_neighbors:
                neighbor.neighbors[l].append(new_node)
                if len(neighbor.neighbors[l]) > self.Mmax:
                    farthest = max(neighbor.neighbors[l], key=lambda x: self.euclidean_distance(x.data, new_node.data))
                    neighbor.neighbors[l].remove(farthest)

        self.nodes.append(new_node)
        if new_node.max_layer > self.entry_point.max_layer:
            self.entry_point = new_node

    def search_layer(self, query_data, ef, layer):
        if self.entry_point is None:
            return []
        W = [self.entry_point]
        visited = set([self.entry_point.id])
        while W:
            c = min(W, key=lambda node: self.euclidean_distance(query_data, node.data))
            W = [n for n in c.neighbors[layer] if n.id not in visited and len(W) < ef]
            visited.update([n.id for n in W])
        return sorted(W, key=lambda node: self.euclidean_distance(query_data, node.data))[:ef]

    def add_point(self, data, id, doc_id=None):
        node_layer = self._get_layer()
        new_node = Node(data, id, node_layer, doc_id=doc_id)
        self._insert_node(new_node)
        if self.entry_point is None or node_layer > self.entry_point.max_layer:
            self.entry_point = new_node

    def search_knn(self, query_data, k, ef=10):
        query_node = Node(query_data, -1, 0)
        candidates = self.search_layer(query_data, ef, 0)
        return sorted(candidates, key=lambda node: self.euclidean_distance(query_node.data, node.data))[:k]

    def save_index(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_index(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

# Example usage:
np.random.seed(42)
dim = 100
embeddings = np.random.rand(100, dim)
hnsw = HNSW(initial_dataset_size=100)
for i, emb in enumerate(embeddings):
    hnsw.add_point(emb, i)
query_embedding = np.random.rand(dim)
nearest_indices = hnsw.search_knn(query_embedding, k=5)
print("Nearest indices:", nearest_indices)
