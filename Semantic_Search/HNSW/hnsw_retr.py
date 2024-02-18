import os
import numpy as np
from sentence_transformers import SentenceTransformer
from Semantic_Search.HNSW.hnsw import HNSW
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
# nltk.download('punkt') # Uncomment if you haven't downloaded NLTK tokenizers
# nltk.download('stopwords') # Uncomment if you haven't downloaded NLTK stopwords

# Assume 'model' is already defined and loaded elsewhere
# from your_model import model

class HNSWTextRetrieval(HNSW):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id_to_text = {}  # Maps node IDs back to text chunks
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def preprocess_text(self, text):
        # Convert text to lowercase
        text = text.lower()
        # Remove special characters and punctuation
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        # Join tokens back into a string
        preprocessed_text = ' '.join(filtered_tokens)
        return preprocessed_text

    def load_and_encode_texts(self, chunk_dir, doc_id):
        directory = os.path.join(chunk_dir, doc_id)
        for i, filename in enumerate(os.listdir(directory)):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                preprocessed_text = self.preprocess_text(text)
                embedding = self.model.encode(preprocessed_text)  # Generate embedding
                node_id = self.insert(embedding)  # Insert embedding into HNSW graph
                self.id_to_text[node_id] = (doc_id, i)  # Store the original text
    
    def update_index(self, chunk_dir):
        for doc_id in os.listdir(chunk_dir):
            path = os.path.join(chunk_dir, doc_id)
            if os.path.isdir(path):
                self.load_and_encode_texts(chunk_dir=chunk_dir, doc_id=doc_id)

    def search(self, query_text, K=5, ef=10, chunk_dir=None):
        # print(query_text)
        query_preprocessed = self.preprocess_text(query_text)
        query_vector = self.model.encode(query_preprocessed)  # Convert query to embedding
        nearest_neighbor_ids = self.k_nn_search(query_vector, K, ef)  # Search in HNSW
        nearest_texts = [self.get_text_from_node(self.id_to_text[n_id], chunk_dir) for n_id in nearest_neighbor_ids]  # Retrieve texts
        return nearest_texts
    
    def get_text_from_node(self, node, chunk_dir):
        with open(f"{os.path.join(chunk_dir, node[0])}/chunk_{node[1]}.txt", "r", encoding="utf-8") as chunk_file:
            return self.preprocess_text(chunk_file.read())
        
    def reset_index(self):
        # Clear existing index data structures
        self.node_data = {}
        self.nodes = []
        self.levels = {}
        self.edges = {}
        self.enter_point = None
        self.max_level = -1
        self.next_id = 0
        self.id_to_text = {}  # Clear the mapping of node IDs to text chunks

    def save_index(self, filepath):
        """Save the HNSW index and text mapping to a file."""
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
        """Load the HNSW index and text mapping from a file."""
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