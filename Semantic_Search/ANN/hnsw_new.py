import numpy as np
from scipy.spatial import distance
import heapq
import os
import pickle

class Node:
    def __init__(self, data, id, doc_id=None):
        self.data = data
        self.id = id
        self.doc_id = doc_id
        self.neighbors = {}  # neighbors per layer

def calculate_distance(a, b):
    return np.linalg.norm(a - b)

class HNSW:
    def __init__(self, M=16):
        self.nodes = {}
        self.entry_point = None
        self.M = M
        self.global_id = 0

    def _get_max_layer(self):
        # Dynamically adjust max_layer based on the current number of nodes
        if len(self.nodes) > 0:
            return max(1, int(np.log2(len(self.nodes))))
        return 0

    def _get_layer(self):
        max_layer = self._get_max_layer()
        layer = 0
        while np.random.random() < 0.5 and layer < max_layer:
            layer += 1
        return layer

    def _select_neighbors(self, node, candidates, layer):
        if layer not in node.neighbors:
            node.neighbors[layer] = []
        distances = [(candidate, calculate_distance(self.nodes[candidate].data, node.data)) for candidate in candidates]
        distances.sort(key=lambda x: x[1])
        selected_neighbors = [candidate[0] for candidate in distances[:self.M]]
        node.neighbors[layer] = selected_neighbors

    def _insert_node(self, node):
        if not self.nodes:
            self.entry_point = node
            self.nodes[node.id] = node
            for i in range(self._get_max_layer() + 1):
                node.neighbors[i] = []
            return

        entry = self.entry_point
        for layer in reversed(range(self._get_max_layer() + 1)):
            entry = self._search_layer(node.data, entry, layer)

        new_node_layer = self._get_layer()
        for layer in range(new_node_layer + 1):
            neighbors = self._search_layer(node.data, entry, layer, search_k=self.M * 2)
            self._select_neighbors(node, neighbors, layer)
            for neighbor_id in neighbors:
                neighbor = self.nodes[neighbor_id]
                self._select_neighbors(neighbor, [node.id] + neighbor.neighbors.get(layer, []), layer)

        self.nodes[node.id] = node
        if new_node_layer > self._get_max_layer():
            self.entry_point = node

    def _search_layer(self, query_data, entry_point, layer, search_k=None):
        if search_k is None:
            search_k = self.M

        visited = set([entry_point.id])  # Ensure entry_point is a Node instance
        candidates = [(0, entry_point)]
        distance_heap = [(calculate_distance(query_data, entry_point.data), entry_point)]

        while candidates:
            _, current_node = heapq.heappop(candidates)
            if current_node.id in visited:
                continue
            visited.add(current_node.id)

            for neighbor_id in current_node.neighbors.get(layer, []):
                neighbor = self.nodes[neighbor_id]
                if neighbor.id not in visited:
                    d = calculate_distance(query_data, neighbor.data)
                    if len(distance_heap) < search_k or d < distance_heap[0][0]:
                        heapq.heappush(distance_heap, (d, neighbor))
                        heapq.heappush(candidates, (d, neighbor))
                        if len(distance_heap) > search_k:
                            heapq.heappop(distance_heap)

        # Return Node instance, not ID
        return sorted(distance_heap, key=lambda x: x[0])[0][1]


    def add_point(self, data, doc_id=None):
        node_id = self.global_id
        self.global_id += 1
        new_node = Node(data, node_id, doc_id)
        self._insert_node(new_node)
        return node_id

    def search_knn(self, query_data, k):
        if not self.nodes:
            return []

        entry = self.entry_point
        for layer in reversed(range(self._get_max_layer() + 1)):
            entry = self._search_layer(query_data, entry, layer)

        candidates = self._search_layer(query_data, entry, 0, search_k=k)
        candidates = [(id, calculate_distance(query_data, self.nodes[id].data)) for id in candidates]
        candidates.sort(key=lambda x: x[1])

        return candidates[:k]

    def get_text_from_node(self, node, chunk_dir):
        with open(f"{os.path.join(chunk_dir, node.doc_id)}/chunk_{node.id}.txt", "r", encoding="utf-8") as chunk_file:
            return chunk_file.read()
            
    def update_index_from_doc_dir(self, chunk_dir, doc_id, model):
        directory_path = os.path.join(chunk_dir, doc_id)
        for i, filename in enumerate(os.listdir(directory_path)):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    emb = model.encode(text)
                    print(doc_id)
                    self.add_point(emb, doc_id=doc_id)
        return 
    
    def update_index_from_chunk_dir(self, chunk_dir):
        for doc_id in os.listdir(chunk_dir):
            path = os.path.join(chunk_dir, doc_id)
            if os.path.isdir(path):
                self.update_index_from_doc_dir(doc_id=doc_id)

    def save_index(self, filename):
        with open(filename, 'wb') as file:
            # Directly use self to refer to the current HNSW index instance
            pickle.dump(self, file)
                        
    @classmethod
    def load_index(cls, filename='hnsw_index.pkl'):
        with open(filename, 'rb') as file:
            loaded_index = pickle.load(file)
        print(f"Index loaded from {filename}.")
        return loaded_index