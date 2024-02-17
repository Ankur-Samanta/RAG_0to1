
import numpy as np
from itertools import product

from ann1 import SimpleANN
class ImprovedANN(SimpleANN):
    def _get_neighboring_bins(self, bin_assignment):
        """Generate neighboring bins to a given bin."""
        neighbors = []
        for dim_shift in product([-1, 0, 1], repeat=len(bin_assignment)):  # Adjust for multi-dimensional
            neighbor_bin = tuple(np.array(bin_assignment) + np.array(dim_shift))
            neighbors.append(neighbor_bin)
        return neighbors

    def search(self, query_embedding, k=1):
        query_bin = self._assign_to_bin(query_embedding)
        candidate_bins = self._get_neighboring_bins(query_bin)
        
        candidates = []
        for bin in candidate_bins:
            candidates.extend(self.grid.get(bin, []))
        
        # Vectorized distance computation
        if candidates:
            candidate_points = np.array([self.points[i] for i in candidates])
            distances = np.linalg.norm(candidate_points - query_embedding, axis=1)
            
            # Get top k results
            nearest_indices = np.argsort(distances)[:k]
            return [candidates[i] for i in nearest_indices], distances[nearest_indices]
        else:
            return [], []

# Helper function
from itertools import product

# Usage remains the same as before, but with the improved search capability
# Generate some random embeddings
np.random.seed(42)
embeddings = np.random.rand(100, 5)  # 100 points in 5-dimensional space

# Index embeddings
ann = ImprovedANN(n_bins=5)
ann.index(embeddings)

# Perform a search
query_embedding = np.random.rand(5)  # Random query in the same space
nearest_indices, distances = ann.search(query_embedding, k=5)

print(f"Nearest indices: {nearest_indices}")
print(f"Distances: {distances}")
