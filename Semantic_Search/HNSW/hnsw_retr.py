import os
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle

from Semantic_Search.HNSW.hnsw import HNSW

# nltk.download('punkt') # Uncomment if you haven't downloaded NLTK tokenizers
# nltk.download('stopwords') # Uncomment if you haven't downloaded NLTK stopwords

class HNSWTextRetrieval(HNSW):
    """
    Parent class for HNSW Index
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the HNSWTextRetrieval class by inheriting from the HNSW class.
        Additionally, initializes a Sentence Transformer model for encoding text into embeddings.
        """
        super().__init__(*args, **kwargs)
        self.id_to_text = {}  # Maps node IDs to their corresponding text chunks
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Initialize the sentence transformer model

    def preprocess_text(self, text):
        """
        Preprocesses the text by converting to lowercase, removing punctuation,
        tokenizing, removing stopwords, and then rejoining into a cleaned string.

        Parameters:
        - text (str): The text to preprocess.

        Returns:
        - str: The preprocessed text.
        """
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\W', ' ', text)  # Remove special characters and punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
        tokens = word_tokenize(text)  # Tokenize
        stop_words = set(stopwords.words('english'))  # Set of stopwords
        filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        preprocessed_text = ' '.join(filtered_tokens)  # Rejoin tokens into a string
        return preprocessed_text

    def load_and_encode_texts(self, chunk_dir, doc_id):
        """
        Loads text chunks from a specified directory, preprocesses, and encodes them into embeddings.
        Each embedding is then inserted into the HNSW graph, and a mapping from node ID to text chunk is stored.

        Parameters:
        - chunk_dir (str): The directory containing text chunks.
        - doc_id (str): The document identifier used for organizing text chunks.
        """
        directory = os.path.join(chunk_dir, doc_id)
        for i, filename in enumerate(os.listdir(directory)):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                preprocessed_text = self.preprocess_text(text)
                embedding = self.model.encode(preprocessed_text)
                node_id = self.insert(embedding)
                self.id_to_text[node_id] = (doc_id, i)

    def update_index(self, chunk_dir):
        """
        Updates the HNSW index with text chunks from all documents in the given directory.

        Parameters:
        - chunk_dir (str): The directory containing folders of text chunks, each folder representing a document.
        """
        for doc_id in os.listdir(chunk_dir):
            path = os.path.join(chunk_dir, doc_id)
            if os.path.isdir(path):
                self.load_and_encode_texts(chunk_dir, doc_id)

    def search(self, query_text, K=5, ef=10, chunk_dir=None):
        """
        Searches the HNSW graph for the nearest text chunks to a given query.

        Parameters:
        - query_text (str): The query text to search for.
        - K (int): The number of nearest neighbors to retrieve.
        - ef (int): The size of the dynamic candidate list.
        - chunk_dir (str): The directory containing text chunks for lookup.

        Returns:
        - list: A list of nearest text chunks to the query.
        """
        query_preprocessed = self.preprocess_text(query_text)
        query_vector = self.model.encode(query_preprocessed)
        nearest_neighbor_ids = self.k_nn_search(query_vector, K, ef)
        nearest_texts = [self.get_text_from_node(self.id_to_text[n_id], chunk_dir) for n_id in nearest_neighbor_ids]
        return nearest_texts

    def get_text_from_node(self, node, chunk_dir):
        """
        Retrieves the text for a given node from the file system using the node's document ID and chunk index.

        Parameters:
        - node (tuple): A tuple containing the document ID and chunk index.
        - chunk_dir (str): The directory containing text chunks for lookup.

        Returns:
        - str: The preprocessed text of the retrieved chunk.
        """
        with open(f"{os.path.join(chunk_dir, node[0])}/chunk_{node[1]}.txt", "r", encoding="utf-8") as chunk_file:
            return self.preprocess_text(chunk_file.read())
        
    def reset_index(self):
        """
        Resets the HNSW graph to an empty state, clearing all stored node data, edges, and text mappings.
        """
        self.node_data = {}
        self.nodes = []
        self.levels = {}
        self.edges = {}
        self.enter_point = None
        self.max_level = -1
        self.next_id = 0
        self.id_to_text = {}  # Clear the mapping of node IDs to text chunks

    def save_index(self, filepath):
        """
        Saves the current state of the HNSW graph and the node ID to text mapping to a file.

        Parameters:
        - filepath (str): The file path where the index should be saved.
        """
        data_to_save = {
            "node_data": self.node_data,
            "nodes": self.nodes,
            "levels": self.levels,
            "edges": self.edges,
            "enter_point": self.enter_point,
            "max_level": self.max_level,
            "next_id": self.next_id,
            "id_to_text": self.id_to_text,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)

    def load_index(self, filepath):
        """
        Loads the HNSW graph and the node ID to text mapping from a file, replacing the current index's state.

        Parameters:
        - filepath (str): The file path from which the index should be loaded.
        """
        with open(filepath, 'rb') as f:
            data_loaded = pickle.load(f)
            self.node_data = data_loaded["node_data"]
            self.nodes = data_loaded["nodes"]
            self.levels = data_loaded["levels"]
            self.edges = data_loaded["edges"]
            self.enter_point = data_loaded["enter_point"]
            self.max_level = data_loaded["max_level"]
            self.next_id = data_loaded["next_id"]
            self.id_to_text = data_loaded["id_to_text"]

# # Example usage
# hnsw_text_retrieval = HNSWTextRetrieval(M=10, Mmax=15, efConstruction=200, mL=1.0)

# chunk_dir = "Data/text_chunks/"

# hnsw_text_retrieval.update_index(chunk_dir)
# hnsw_text_retrieval.reset_index(chunk_dir)
# hnsw_text_retrieval.update_index(chunk_dir)


# query_text = "reliable individual level neural markers high level language processing necessary precursor relating neural variability behavioral genetic variability neuroimage 139 74 93 2016 192 fedorenko e thompson schill l"
# nearest_texts = hnsw_text_retrieval.search(query_text, K=5, ef=10, chunk_dir=chunk_dir)
# for text in nearest_texts:
#     print(text)
#     print("\n")