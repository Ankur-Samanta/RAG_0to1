import numpy as np
from scipy.spatial import distance
import math
import pickle
import os

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
        self.initial_dataset_size = initial_dataset_size
        self.reset_index()
        self.reindex_threshold = 1.5  # Trigger re-index when size is 150% of the initial or last re-indexed size

    def update_parameters(self, new_dataset_size):
        self.dataset_size = new_dataset_size
        self.max_layer = self.calculate_max_layer(new_dataset_size)
        # Optionally adjust M based on new insights or benchmarking
        self.M = self.calculate_M()  # Or a new method to calculate M dynamically

    def calculate_max_layer(self, dataset_size):
        return int(np.log2(dataset_size))  # Example rule, adjust as needed

    def calculate_M(self):
        # Example dynamic calculation of M based on dataset size
        # This is a placeholder: Adjust the formula according to your requirements
        return max(30, min(30, int(np.sqrt(self.dataset_size))))

    def reset_index(self):
        """Resets the HNSW index to its initial state."""
        self.nodes = []
        self.entry_point = None
        self.dataset_size = self.initial_dataset_size
        self.max_layer = self.calculate_max_layer(self.initial_dataset_size)
        self.M = self.calculate_M()
        self.node_map = {}  # Resetting the map of unique keys to node references
        
    def reindex(self):
        # Infer new dataset size from the length of unique node keys
        new_dataset_size = len(self.node_map)
        
        # Store current data points
        # Assuming data, id, and doc_id are enough to reconstruct each node
        data_points = [(node.data, node.id, node.doc_id) for node in self.nodes]
        
        # Reset the index to clear the existing structure
        self.reset_index()
        
        # Update parameters based on the new dataset size
        self.update_parameters(new_dataset_size)
        
        # Re-add all the data points to the index with updated parameters
        for data, id, doc_id in data_points:
            self.add_point(data, id, doc_id)
    
    def euclidean_distance(self, a, b):
        return distance.euclidean(a.data, b.data)

    def _get_layer(self):
        # Probabilistic layer assignment. Adjust p value as needed.
        p = 0.5
        layer = 0
        while np.random.rand() < p and layer < self.max_layer:
            layer += 1
        return layer
    
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
        unique_key = (doc_id, id)
        
        # Check if the node already exists in the node_map
        if unique_key in self.node_map:
            # print(f"Node with doc_id={doc_id}, id={id} already exists. Updating data.")
            existing_node = self.node_map[unique_key]
            existing_node.data = data
            # Perform any additional updates required
        else:
            node_layer = self._get_layer()
            new_node = Node(data, id, node_layer, doc_id=doc_id)
            self._insert_node(new_node)
            
            # Update entry point if necessary
            if self.entry_point is None or node_layer > self.entry_point.max_layer:
                self.entry_point = new_node
            
            # Add the new node reference to both self.nodes and self.node_map
            self.nodes.append(new_node)
            self.node_map[unique_key] = new_node
            
            # Check if re-indexing is needed
            if len(self.node_map) > self.reindex_threshold * self.dataset_size:
                print("Triggering re-index.")
                self.reindex()

        
    def search_knn(self, query_data, k, ef=10):
        if not self.entry_point:
            return []  # Handle case with no entry point
        query_node = Node(query_data, -1, 0, None)
        current_node = self.entry_point
        visited = set([self.entry_point.id])  # Initialize with entry point ID
        candidates = []

        for l in range(min(self.max_layer, current_node.max_layer), -1, -1):  # Adjusted to handle dynamic layers
            while True:
                closer_node = None
                min_distance = float('inf')
                for neighbor in current_node.neighbors.get(l, []):
                    if neighbor.id not in visited:
                        distance = self.euclidean_distance(query_node, neighbor)
                        if distance < min_distance:
                            closer_node = neighbor
                            min_distance = distance
                if closer_node:
                    current_node = closer_node
                    visited.add(closer_node.id)
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
            
    def update_index_from_doc_dir(self, chunk_dir, doc_id, model):
        directory_path = os.path.join(chunk_dir, doc_id)
        for i, filename in enumerate(os.listdir(directory_path)):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    emb = model.encode(text)
                    self.add_point(emb, i, doc_id=doc_id)
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