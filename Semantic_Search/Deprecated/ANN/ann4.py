from sklearn.cluster import KMeans
import numpy as np
from itertools import product
from sklearn.metrics.pairwise import euclidean_distances
import os

class AdaptiveGridANN:
    def __init__(self, embeddings, n_clusters=None, n_quantiles=10, predefined_bins=None):
        self.embeddings = embeddings
        self.n_clusters = n_clusters
        self.n_quantiles = n_quantiles
        self.predefined_bins = predefined_bins
        self.grid = {}
        self.points = []
        
        if predefined_bins is not None:
            self.bins_per_dimension = predefined_bins
        else:
            self.labels, _ = self._cluster_data(embeddings, n_clusters)
            self.quantiles = self._calculate_quantiles_per_cluster(embeddings, self.labels, n_clusters, n_quantiles)
            self.bins_per_dimension = [n_quantiles - 1] * embeddings.shape[1]  # Assuming uniform quantile bins across dimensions

    def _cluster_data(self, embeddings, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
        return kmeans.labels_, kmeans.cluster_centers_

    def _calculate_quantiles_per_cluster(self, embeddings, labels, n_clusters, n_quantiles):
        quantiles = {}
        for cluster_id in range(n_clusters):
            cluster_points = embeddings[labels == cluster_id]
            quantiles[cluster_id] = np.percentile(cluster_points, np.linspace(0, 100, n_quantiles), axis=0)
        return quantiles

    def _assign_to_bin(self, point, cluster_id=None):
        if self.predefined_bins is not None:
            # Use predefined bins
            bin_assignment = tuple((point * self.predefined_bins).astype(int))
        else:
            # Use adaptive bins based on quantiles
            quantile_ranges = self.quantiles[cluster_id]
            bin_assignment = []
            for dim in range(point.shape[0]):
                q_index = np.digitize(point[dim], quantile_ranges[:, dim]) - 1
                bin_assignment.append(q_index)
            bin_assignment = tuple(bin_assignment)
        return bin_assignment

    def index(self, embeddings):
        for i, point in enumerate(embeddings):
            self.points.append(point)
            if self.predefined_bins is None:
                cluster_id = self.labels[i]
                bin_assignment = self._assign_to_bin(point, cluster_id)
            else:
                bin_assignment = self._assign_to_bin(point)
            
            if bin_assignment not in self.grid:
                self.grid[bin_assignment] = []
            self.grid[bin_assignment].append(i)

    def search(self, query_embedding, k=1):
        # Initialize a set to collect candidate indices from the appropriate bins
        candidate_indices = set()
        
        if self.predefined_bins is not None:
            query_bin = self._assign_to_bin(query_embedding)
            # Directly add candidates from the query bin if it exists in the grid
            if query_bin in self.grid:
                candidate_indices.update(self.grid[query_bin])
        else:
            # For adaptive bins, consider candidates from bins corresponding to all clusters
            for cluster_id in range(self.n_clusters):
                query_bin = self._assign_to_bin(query_embedding, cluster_id)
                neighboring_bins = self._get_neighboring_bins(query_bin, cluster_id)
                for bin in neighboring_bins:
                    if bin in self.grid:
                        candidate_indices.update(self.grid[bin])
        
        # If no candidates are found, return empty lists
        if not candidate_indices:
            return [], []

        # Calculate distances from the query point to each candidate
        candidate_points = np.array([self.points[i] for i in candidate_indices])
        distances = euclidean_distances([query_embedding], candidate_points).flatten()
        
        # Get the indices of the k smallest distances
        nearest_indices = np.argsort(distances)[:k]
        selected_candidates = np.array(list(candidate_indices))[nearest_indices]
        selected_distances = distances[nearest_indices]

        return selected_candidates.tolist(), selected_distances.tolist()

    def _get_neighboring_bins(self, query_bin, cluster_id):
        # Assuming a simple neighboring strategy that checks immediate neighbors in all dimensions
        neighbors = []
        for dim_shift in product([-1, 0, 1], repeat=len(query_bin)):  # Adjust based on dimensionality
            neighbor_bin = tuple(np.array(query_bin) + np.array(dim_shift))
            # Make sure the neighbor bin is valid (within grid bounds and quantiles if adaptive)
            if self._is_valid_bin(neighbor_bin, cluster_id):
                neighbors.append(neighbor_bin)
        return neighbors

    def _is_valid_bin(self, bin, cluster_id):
        # Implement logic to check if a bin is valid based on predefined bins or quantiles for adaptive bins
        # This is a placeholder function; you'll need to adapt it based on your grid structure and data distribution
        return True  # Placeholder
    
    def update_index_from_chunk_dir(self, directory_path, doc_id, model):
        directory_path = os.path.join(directory_path, doc_id)
        documents = []
        for i, filename in enumerate(os.listdir(directory_path)):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    documents.append(text)
        embeddings = model.encode(documents)
        return embeddings
                    

# Example usage
np.random.seed(42)
dim=768
embeddings = np.random.rand(350, dim)  # 100 points in a 5-dimensional space
# Initialize with predefined bins
#ann_predefined = AdaptiveGridANN(embeddings, predefined_bins=np.array([10, 10, 10, 10, 10]))

# Initialize with adaptive bins through clustering
ann_adaptive = AdaptiveGridANN(embeddings, n_clusters=100, n_quantiles=10) #10
ann_adaptive.index(embeddings)

print("Start")

# Query
query_embedding = np.random.rand(dim)
k = 5
nearest_indices, distances = ann_adaptive.search(query_embedding, k)

print(f"Nearest indices: {nearest_indices}")
print(f"Distances: {distances}")
