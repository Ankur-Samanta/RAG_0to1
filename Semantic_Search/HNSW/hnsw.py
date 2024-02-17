import numpy as np
import math
import heapq
import pickle
import os

class HNSW:
    def __init__(self, M=10, Mmax=15, efConstruction=200, mL=1.0):
        self.M = M  # Max number of connections for each element per layer
        self.Mmax = Mmax  # Maximum number of connections for each element per layer
        self.efConstruction = efConstruction  # Size of the dynamic candidate list
        self.mL = mL  # Normalization factor for level generation
        self.node_data = {}  # Key: node ID, Value: node data (numpy.ndarray)
        self.nodes = []  # List of node IDs
        self.levels = {}  # Key: node ID, Value: level
        self.edges = {}  # Key: node ID, Value: dict of edges per node per level
        self.enter_point = None
        self.max_level = -1
        self.next_id = 0  # To generate unique IDs

    def _add_node(self, node_id, level, neighbors):
        # Initialize node's levels and edges if not already done
        self.levels[node_id] = level
        if node_id not in self.edges:
            self.edges[node_id] = {i: set() for i in range(level + 1)}
        else:
            # Ensure all levels up to 'level' are initialized for node_id
            for i in range(level + 1):
                if i not in self.edges[node_id]:
                    self.edges[node_id][i] = set()

        for lc in range(level + 1):
            for neighbor_id in neighbors:
                # Ensure neighbor's levels and edges are initialized
                if neighbor_id not in self.levels:
                    self.levels[neighbor_id] = 0  # Default to level 0 if not set, adjust as necessary
                if neighbor_id not in self.edges:
                    self.edges[neighbor_id] = {0: set()}
                
                # Ensure all levels up to 'lc' are initialized for neighbor_id
                for i in range(lc + 1):
                    if i not in self.edges[neighbor_id]:
                        self.edges[neighbor_id][i] = set()

                # Now safe to add connection
                self.edges[node_id][lc].add(neighbor_id)
                self.edges[neighbor_id][lc].add(node_id)
                #print(f"Level {lc}: Node {node_id} <-> Node {neighbor_id}")


    def _random_level(self):
        lvl = int(-math.log(np.random.uniform()) * self.mL)
        return lvl

    def _distance(self, a_id, b_id):
        a = self.node_data[a_id]
        b = self.node_data[b_id]
        return np.linalg.norm(a-b)


    def _search_layer(self, q_id, ep_id, ef, lc):
        v = set()  # Visited nodes
        C = []  # Candidates as a min heap
        heapq.heappush(C, (0, ep_id))  # Distance to itself is 0

        W = []  # Nearest neighbors found

        while C:
            dist, current_id = heapq.heappop(C)
            if current_id in v:
                continue
            v.add(current_id)

            if len(W) < ef or dist < W[0][0]:
                heapq.heappush(W, (-dist, current_id))  # Use negative distance because heapq is a min heap
                if len(W) > ef:
                    heapq.heappop(W)  # Keep only the ef closest

            for neighbor_id in self.edges[current_id].get(lc, []):
                if neighbor_id not in v:
                    d = self._distance(q_id, neighbor_id)
                    heapq.heappush(C, (d, neighbor_id))

        nearest_ids = [id for _, id in sorted([(-dist, id) for dist, id in W], reverse=True)]
        # print(f"Visiting: {current_id}, Distance: {dist}")
        # print(f"Nearest IDs: {nearest_ids}")
        return nearest_ids



    def _select_neighbors_simple(self, q_id, C_ids, M):
        #print("C_ids", C_ids)
        distances = [(c_id, self._distance(q_id, c_id)) for c_id in C_ids]
        #print("dist", distances)
        distances.sort(key=lambda x: x[1])
        selected_ids = [x[0] for x in distances[:M]]
        #print("sel-d", selected_ids)
        return selected_ids
    
    def insert(self, point_data):
        node_id = self.next_id
        self.next_id += 1
        self.node_data[node_id] = point_data
        self.nodes.append(node_id)  # Add this line to include the node ID in the nodes list

        if not self.enter_point:
            self.levels[node_id] = 0
            self.edges[node_id] = {0: set()}
            self.enter_point = node_id
            self.max_level = 0
            return node_id

        l = self._random_level()
        
        # Ensure that all levels are initialized for the new node up to its level, including intermediate levels
        self.levels[node_id] = l
        if node_id not in self.edges:
            self.edges[node_id] = {}
        for i in range(l + 1):
            if i not in self.edges[node_id]:
                self.edges[node_id][i] = set()
        
        if l > self.max_level:
            self.max_level = l
            self.enter_point = node_id

        ep = self.enter_point
        L = self.max_level

        for lc in range(L, l, -1):
            W = self._search_layer(node_id, ep, 1, lc)
            if W:
                ep = W[0]

        for lc in range(min(L, l), -1, -1):
            W = self._search_layer(node_id, ep, self.efConstruction, lc)
            neighbors = self._select_neighbors_simple(node_id, W, self.M if lc < l else self.Mmax)
            self._add_node(node_id, lc, neighbors)

            for neighbor_id in neighbors:
                all_neighbors = list(self.edges[neighbor_id].get(lc, set()))
                if len(all_neighbors) > self.Mmax:
                    new_neighbors = self._select_neighbors_simple(neighbor_id, all_neighbors, self.Mmax)
                    self.edges[neighbor_id][lc] = set(new_neighbors)
                    for n in all_neighbors:
                        if n not in new_neighbors:
                            self.edges[n][lc].discard(neighbor_id)


        # Update the entry point if the new node's level is higher
        if l > L:
            self.enter_point = node_id
            
        return node_id


    def k_nn_search(self, q_data, K, ef):
        # print(self.nodes)
        # Check if there are any nodes in the graph
        if not self.nodes:
            return []

        # Temporarily add query data for distance calculations
        q_id = self.next_id
        self.node_data[q_id] = q_data
        # Get the entry point and the top layer level
        ep = self.enter_point
        L = self.levels[ep] if self.enter_point else -1
        # Initialize the candidate set for the nearest neighbors
        candidates = []
        try:
            # Start from the top layer and move down
            for lc in range(L, 0, -1):  # Iterate down to layer 1
                temp_candidates = self._search_layer(q_id, ep, 1, lc)
                if temp_candidates:
                    ep = temp_candidates[0]  # Update entry point for the next layer

            # Perform the search on the base layer
            candidates = self._search_layer(q_id, ep, ef, lc=0)
            # Refine to the top K nearest neighbors
            distances = [(n_id, self._distance(q_id, n_id)) for n_id in candidates]
            distances.sort(key=lambda x: x[1])
            nearest_neighbor_ids = [n_id for n_id, _ in distances[:K]]

            # Get the data for the nearest neighbors
            nearest_neighbors = [self.node_data[n_id] for n_id in nearest_neighbor_ids]
            nearest_neighbors = nearest_neighbor_ids


        finally:
            # Cleanup: remove the temporary query data
            del self.node_data[q_id]
            self.next_id -= 1

        return nearest_neighbors
    
    # def get_text_from_node(self, node, chunk_dir):
    #     with open(f"{os.path.join(chunk_dir, node.doc_id)}/chunk_{node.id}.txt", "r", encoding="utf-8") as chunk_file:
    #         return chunk_file.read()
            
    # def update_index_from_doc_dir(self, chunk_dir, doc_id):
    #     directory_path = os.path.join(chunk_dir, doc_id)
    #     for i, filename in enumerate(os.listdir(directory_path)):
    #         file_path = os.path.join(directory_path, filename)
    #         if os.path.isfile(file_path):
    #             with open(file_path, 'r', encoding='utf-8') as file:
    #                 text = file.read()
    #                 emb = self.model.encode(text)
    #                 self.add_point(emb, i, doc_id=doc_id)
    #     return 
    
    # def update_index_from_chunk_dir(self, chunk_dir):
    #     for doc_id in os.listdir(chunk_dir):
    #         path = os.path.join(chunk_dir, doc_id)
    #         if os.path.isdir(path):
    #             self.update_index_from_doc_dir(chunk_dir=chunk_dir, doc_id=doc_id)

    # def save_index(self, filename):
    #     with open(filename, 'wb') as file:
    #         # Directly use self to refer to the current HNSW index instance
    #         pickle.dump(self, file)
                        
    # @classmethod
    # def load_index(cls, filename='hnsw_index.pkl'):
    #     with open(filename, 'rb') as file:
    #         loaded_index = pickle.load(file)
    #     print(f"Index loaded from {filename}.")
    #     return loaded_index


# # Example usage
# hnsw = HNSW(M=5, Mmax=10, efConstruction=50, mL=25)
# data = np.random.rand(50, 768)  # 100 5-dimensional points

# # Insert data points into the HNSW graph
# for point in data:
#     hnsw.insert(point)

# # Perform a k-NN search for a random query point
# query_point = np.random.rand(768)  # A random 5-dimensional query point
# K = 5  # Number of nearest neighbors to find
# nearest_neighbors = hnsw.k_nn_search(query_point, K, ef=10)

# # print(f"Query Point: {query_point}")
# print(f"{K} Nearest Neighbors: {nearest_neighbors}")

# # print(hnsw.levels)