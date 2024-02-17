import numpy as np
from itertools import product

class MultiResolutionANN:
    def __init__(self, layers_config=[10, 20, 30]):
        """
        Initialize the ANN with multiple resolution layers.
        layers_config defines the grid size for each layer, with smaller numbers indicating coarser grids.
        """
        self.layers_config = layers_config
        self.layers = [{} for _ in layers_config]  # Grids for each layer
        self.points = []  # To store the actual points
    
    def _assign_to_bin(self, point, grid_size):
        """Assign a point to a bin in a specific grid."""
        # Normalize point values to [0, grid_size] for bin assignment
        return tuple((point * grid_size).astype(int))
    
    def _get_neighboring_bins(self, bin_assignment, layer_index):
        """Generate neighboring bins for a given bin."""
        neighbors = []
        grid_size = self.layers_config[layer_index]
        for dim_shift in product([-1, 0, 1], repeat=len(bin_assignment)):
            neighbor_bin = tuple(np.array(bin_assignment) + np.array(dim_shift))
            # Ensure bin is within grid bounds
            if all(0 <= n < grid_size for n in neighbor_bin):
                neighbors.append(neighbor_bin)
        return neighbors

    def index(self, embeddings):
        """Index embeddings across multiple layers of grids."""
        for i, point in enumerate(embeddings):
            self.points.append(point)
            for layer_idx, grid_size in enumerate(self.layers_config):
                bin_assignment = self._assign_to_bin(point, grid_size)
                if bin_assignment not in self.layers[layer_idx]:
                    self.layers[layer_idx][bin_assignment] = []
                self.layers[layer_idx][bin_assignment].append(i)
    
    def search(self, query_embedding, k=1):
        """Perform a search for the k nearest neighbors across layers."""
        candidates = set()
        for layer_idx, grid_size in enumerate(self.layers_config):
            query_bin = self._assign_to_bin(query_embedding, grid_size)
            neighboring_bins = self._get_neighboring_bins(query_bin, layer_idx)
            
            for neighbor_bin in neighboring_bins:
                if neighbor_bin in self.layers[layer_idx]:
                    candidates.update(self.layers[layer_idx][neighbor_bin])
        
        if not candidates:
            return [], []

        # Calculate distances for candidates
        candidate_points = np.array([self.points[i] for i in candidates])
        distances = np.linalg.norm(candidate_points - query_embedding, axis=1)
        
        # Sort by distance and select top k
        nearest_indices = np.argsort(distances)[:k]
        selected_candidates = np.array(list(candidates))[nearest_indices]
        selected_distances = distances[nearest_indices]

        return selected_candidates, selected_distances

# Usage Example
np.random.seed(42)  # For consistent random results
embeddings = np.random.rand(100, 5)  # 100 5-dimensional embeddings

# Initialize and index embeddings
ann = MultiResolutionANN([1, 2, 3])
ann.index(embeddings)

# Query
query_embedding = np.random.rand(5)
k = 5
nearest_indices, distances = ann.search(query_embedding, k)

print(f"Nearest indices: {nearest_indices}")
print(f"Distances: {distances}")
