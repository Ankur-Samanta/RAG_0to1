'''
Note: This is an approximate and simplified implementation of the HNSW (Hierarchical Navigable Small World graphs) approximate nearest neighbor search algorithm:
Paper: https://arxiv.org/pdf/1603.09320.pdf
'''


import numpy as np
import math
import heapq
import pickle
import os

class HNSW:
    def __init__(self, M=10, Mmax=15, efConstruction=200, mL=1.0):
        """
        Initialize the HNSW graph with default or user-defined parameters.

        Parameters:
        - M (int): The maximum number of outgoing connections in the layer of the graph.
        - Mmax (int): The maximum number of connections for each element at the base layer.
        - efConstruction (int): The size of the dynamic candidate list during insertion/search.
        - mL (float): The level generation factor, influences the distribution of elements across layers.
        """
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
        """
        Adds a node and its connections to the graph.

        Parameters:
        - node_id (int): The ID of the node being added.
        - level (int): The level of the node.
        - neighbors (list): A list of neighbor node IDs to connect with.
        """
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
        """
        Generates a random level for a new node based on the level generation factor (mL).

        Returns:
        - int: The randomly selected level for the new node.
        """
        lvl = int(-math.log(np.random.uniform()) * self.mL)
        return lvl

    def _distance(self, a_id, b_id):
        """
        Computes the Euclidean distance between two nodes based on their data (vector embeddings).

        Parameters:
        - a_id (int): Node ID of the first node.
        - b_id (int): Node ID of the second node.

        Returns:
        - float: The Euclidean distance between the two nodes.
        """
        a = self.node_data[a_id]
        b = self.node_data[b_id]
        return np.linalg.norm(a-b)


    def _search_layer(self, q_id, ep_id, ef, lc):
        """
        Searches for the nearest neighbors of a query node in a specific layer.

        Parameters:
        - q_id (int): The query node ID.
        - ep_id (int): The entry point node ID for the search.
        - ef (int): The size of the dynamic candidate list.
        - lc (int): The layer at which the search is conducted.

        Returns:
        - list: A list of the nearest neighbor IDs.
        """
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
        """
        Selects the top M nearest neighbors from a candidate list based on their distances to the query node.

        Parameters:
        - q_id (int): The query node ID.
        - C_ids (list): A list of candidate node IDs.
        - M (int): The number of neighbors to select.

        Returns:
        - list: A list of the selected neighbor IDs.
        """
        #print("C_ids", C_ids)
        distances = [(c_id, self._distance(q_id, c_id)) for c_id in C_ids]
        #print("dist", distances)
        distances.sort(key=lambda x: x[1])
        selected_ids = [x[0] for x in distances[:M]]
        #print("sel-d", selected_ids)
        return selected_ids
    
    def insert(self, point_data):
        """
        Inserts a new node with the given data into the graph. This method automatically assigns a unique ID
        to the new node, determines its level based on the probabilistic distribution, and establishes connections
        to existing nodes in the graph to maintain the HNSW structure.

        Parameters:
        - point_data (np.ndarray): The data (vector embedding) associated with the node to be inserted.

        Returns:
        - int: The unique ID assigned to the newly inserted node.
        """
        # Assign a unique ID to the new node and store its data.
        node_id = self.next_id
        self.next_id += 1  # Increment the ID counter for the next node.
        self.node_data[node_id] = point_data
        self.nodes.append(node_id)  # Track the new node's ID.

        # If this is the first node, initialize the graph with this node as the entry point.
        if not self.enter_point:
            self.levels[node_id] = 0  # The first node starts at level 0.
            self.edges[node_id] = {0: set()}  # Initialize its edges with an empty set at level 0.
            self.enter_point = node_id  # Set this node as the entry point for the graph.
            self.max_level = 0  # This is currently the highest level in the graph.
            return node_id  # Return the ID of the first node.

        # Determine the level of the new node based on a random distribution.
        l = self._random_level()

        # Initialize the node's level and edges structure.
        self.levels[node_id] = l
        if node_id not in self.edges:
            self.edges[node_id] = {}
        for i in range(l + 1):
            if i not in self.edges[node_id]:
                self.edges[node_id][i] = set()

        # If the node's level is higher than the current max level, update the graph's max level and entry point.
        if l > self.max_level:
            self.max_level = l
            self.enter_point = node_id

        # Start the connection process from the current entry point and max level.
        ep = self.enter_point
        L = self.max_level

        # For each level above the new node's level, find the closest node to update the entry point.
        for lc in range(L, l, -1):
            W = self._search_layer(node_id, ep, 1, lc)
            if W:  # Update the entry point for the next lower level if neighbors are found.
                ep = W[0]

        # Connect the new node to its neighbors at each level from its level down to 0.
        for lc in range(min(L, l), -1, -1):
            W = self._search_layer(node_id, ep, self.efConstruction, lc)
            neighbors = self._select_neighbors_simple(node_id, W, self.M if lc < l else self.Mmax)
            self._add_node(node_id, lc, neighbors)

            # If any neighbor has more connections than allowed, select the best connections based on distance.
            for neighbor_id in neighbors:
                all_neighbors = list(self.edges[neighbor_id].get(lc, set()))
                if len(all_neighbors) > self.Mmax:
                    new_neighbors = self._select_neighbors_simple(neighbor_id, all_neighbors, self.Mmax)
                    self.edges[neighbor_id][lc] = set(new_neighbors)
                    for n in all_neighbors:
                        if n not in new_neighbors:
                            self.edges[n][lc].discard(neighbor_id)

        # If the new node's level is higher than the graph's current max level, update the entry point.
        if l > L:
            self.enter_point = node_id

        return node_id  # Return the ID of the newly inserted node.



    def k_nn_search(self, q_data, K, ef):
        """
        Performs a k-nearest neighbor search for the given query data.

        Parameters:
        - q_data (np.ndarray): The query data (vector embedding).
        - K (int): The number of nearest neighbors to find.
        - ef (int): The size of the dynamic candidate list during the search.

        Returns:
        - list: A list of the K nearest neighbor data.
        """
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