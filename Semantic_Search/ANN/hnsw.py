import numpy as np
from scipy.spatial import distance
import math
import pickle
import os
from sentence_transformers import SentenceTransformer


class Node:
    def __init__(self, data, id, max_layer, doc_id=None):
        self.data = data
        self.id = id
        self.max_layer = max_layer
        self.doc_id=doc_id
        # Initialize neighbors for all layers up to the graph's max_layer
        self.neighbors = {i: [] for i in range(max_layer + 1)}



class HNSW:
    def __init__(self, initial_dataset_size):
        self.nodes = []
        self.entry_point = None
        self.dataset_size = initial_dataset_size
        self.max_layer = self.calculate_max_layer(initial_dataset_size)
        self.M = self.calculate_M()  # Define based on empirical testing
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def update_config(self, new_dataset_size):
            self.dataset_size = new_dataset_size
            self.max_layer = self.calculate_max_layer(new_dataset_size)
            
    def calculate_max_layer(self, dataset_size):
        return int(np.log2(dataset_size))  # Example rule, adjust as needed

    def calculate_M(self):
        # Example dynamic calculation of M based on dataset size
        # This is a placeholder: Adjust the formula according to your requirements
        return max(16, min(30, int(np.sqrt(self.dataset_size))))
        
    def euclidean_distance(self, a, b):
        return distance.euclidean(a.data, b.data)
        
    def _get_layer(self):
        mL = 1 / np.log(self.M)
        l = 0
        while np.random.rand() < np.exp(-l / mL) and l < self.max_layer:
            l += 1
        return l
    
    def _select_neighbors_heuristic(self, candidates, M):
        # Simplified heuristic: sort by distance and select top M
        return sorted(candidates, key=lambda x: self.euclidean_distance(x[0], x[1]))[:M]
    
    def _update_neighbors(self, node, candidates, layer):
        # Update connections with a heuristic for best neighbors
        selected_neighbors = self._select_neighbors_heuristic(candidates, self.M)
        node.neighbors[layer] = [n for n, _ in selected_neighbors]
        for neighbor, _ in selected_neighbors:
            neighbor.neighbors[layer].append(node)  # Ensure bi-directional connection
        
    def _insert_node(self, new_node):
        if not self.nodes:  # Handle the first node specially
            self.nodes.append(new_node)
            self.entry_point = new_node
            return

        # Find the entry point for the new node at the highest layer
        entry_point = self.entry_point
        
        # 1. Find the closest nodes in the graph to the new node for each layer
        for l in reversed(range(new_node.max_layer + 1)):
            current_closest = entry_point
            while True:
                changed = False
                # Iterate through neighbors at this layer to find closer one
                for neighbor in current_closest.neighbors.get(l, []):
                    if self.euclidean_distance(new_node, neighbor) < self.euclidean_distance(new_node, current_closest):
                        current_closest = neighbor
                        changed = True
                if not changed:  # If no closer node is found, break the loop
                    break
            # Update entry_point for the next layer down
            entry_point = current_closest

            # 2. Connect the new_node with its closest nodes at layer l
            neighbors_l = [n for n in self.nodes if l in n.neighbors]  # All nodes participating in layer l
            # Calculate distances to all neighbors_l and sort
            distances = [(n, self.euclidean_distance(new_node, n)) for n in neighbors_l]
            distances.sort(key=lambda x: x[1])
            closest_neighbors = [n for n, dist in distances[:self.M]]  # Select top M closest nodes

            # Update neighbors for new_node and reciprocal connections for selected neighbors
            new_node.neighbors[l] = closest_neighbors
            for neighbor in closest_neighbors:
                neighbor.neighbors[l].append(new_node)  # Add new_node as neighbor
                # Ensure we don't exceed M neighbors, remove the farthest if necessary
                if len(neighbor.neighbors[l]) > self.M:
                    farthest = max(neighbor.neighbors[l], key=lambda x: self.euclidean_distance(neighbor, x))
                    neighbor.neighbors[l].remove(farthest)

        self.nodes.append(new_node)  # Add new_node to the graph

        # Optionally, update the global entry point if new_node's layer is higher
        if self.entry_point.max_layer < new_node.max_layer:
            self.entry_point = new_node

    
    def add_point(self, data, id, doc_id=None):
        node_layer = self._get_layer()
        new_node = Node(data, id, node_layer, doc_id=doc_id)
        self._insert_node(new_node)
        if self.entry_point is None or node_layer > self.entry_point.max_layer:
            self.entry_point = new_node
    
    def search_knn(self, query_data, k, ef=10):
        query_node = Node(query_data, -1, 0)
        current_node = self.entry_point
        for l in range(self.max_layer, -1, -1):
            while True:
                closer_node = min(
                    current_node.neighbors[l],
                    key=lambda n: self.euclidean_distance(query_node, n),
                    default=None
                )
                if closer_node and self.euclidean_distance(query_node, closer_node) < self.euclidean_distance(query_node, current_node):
                    current_node = closer_node
                else:
                    break
        
        # Implement the final search in layer 0 with dynamic candidates list (ef)
        candidates = [(node, self.euclidean_distance(query_node, node)) for node in current_node.neighbors[0]]
        candidates = sorted(candidates, key=lambda x: x[1])[:ef]  # Simulate dynamic list with ef
        
        # Return top-k results
        return [n for n, _ in sorted(candidates, key=lambda x: x[1])[:k]]
    
    def get_text_from_node(self, node, chunk_dir):
        with open(f"{os.path.join(chunk_dir, node.doc_id)}/chunk_{node.id}.txt", "r", encoding="utf-8") as chunk_file:
            return chunk_file.read()
            
    def update_index_from_doc_dir(self, chunk_dir, doc_id):
        directory_path = os.path.join(chunk_dir, doc_id)
        for i, filename in enumerate(os.listdir(directory_path)):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    emb = self.model.encode(text)
                    self.add_point(emb, i, doc_id=doc_id)
        return 
    
    def update_index_from_chunk_dir(self, chunk_dir):
        for doc_id in os.listdir(chunk_dir):
            path = os.path.join(chunk_dir, doc_id)
            if os.path.isdir(path):
                self.update_index_from_doc_dir(chunk_dir=chunk_dir, doc_id=doc_id)

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

# # Example usage
# np.random.seed(42)
# dim = 100
# embeddings = np.random.rand(100, dim)  # 100 5-dimensional embeddings
# ann = HNSW(initial_dataset_size=100)
# for i, emb in enumerate(embeddings):
#     ann.add_point(emb, i)

# query_embedding = np.random.rand(dim)
# k = 5
# nearest_indices = ann.search_knn(query_embedding, k)

# print(f"Nearest indices: {nearest_indices}")