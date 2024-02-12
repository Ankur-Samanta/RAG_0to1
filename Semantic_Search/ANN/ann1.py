import numpy as np

class SimpleANN:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins  # Number of bins in each dimension
        self.points = []  # List to store points
        self.grid = {}  # Dictionary to store points' indices by their bin
    
    def _assign_to_bin(self, point):
        # Simplified bin assignment for demonstration
        return tuple((point * self.n_bins).astype(int))
    
    def index(self, embeddings):
        """Index embeddings by assigning them to bins."""
        for i, point in enumerate(embeddings):
            bin_assignment = self._assign_to_bin(point)
            if bin_assignment not in self.grid:
                self.grid[bin_assignment] = []
            self.grid[bin_assignment].append(i)
            self.points.append(point)
    
    def search(self, query_embedding, k=1):
        """Search for the k nearest neighbors of the query_embedding."""
        query_bin = self._assign_to_bin(query_embedding)
        candidates = self.grid.get(query_bin, [])
        
        # Compute distances from the query to each candidate
        distances = [np.linalg.norm(query_embedding - self.points[i]) for i in candidates]
        
        # Find the indices of the k smallest distances
        if distances:
            nearest_indices = np.argsort(distances)[:k]
            return [candidates[i] for i in nearest_indices], [distances[i] for i in nearest_indices]
        else:
            return [], []

# Generate some random embeddings
np.random.seed(42)
embeddings = np.random.rand(100, 5)  # 100 points in 5-dimensional space

# Index embeddings
ann = SimpleANN(n_bins=5)
ann.index(embeddings)

# Perform a search
query_embedding = np.random.rand(5)  # Random query in the same space
nearest_indices, distances = ann.search(query_embedding, k=5)

print(f"Nearest indices: {nearest_indices}")
print(f"Distances: {distances}")
