import faiss
from sentence_transformers import SentenceTransformer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import pickle

class FAISSTextRetrieval:
    def __init__(self, *args, **kwargs):
        """
        Initializes the FAISSTextRetrieval class with a Sentence Transformer model for encoding text.
        Prepares for the creation of a FAISS index for efficient similarity search.
        """
        self.id_to_text = {}  # Maps node IDs to their corresponding text chunks
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Initialize the sentence transformer model
        self.index = None  # Placeholder for FAISS index, to be initialized later
        self.dimension = self.model.get_sentence_embedding_dimension()  # Dimension of embeddings

    def preprocess_text(self, text):
        """
        Preprocesses the text by converting to lowercase, removing punctuation,
        tokenizing, removing stopwords, and then rejoining into a cleaned string.

        Parameters:
        - text (str): The text to preprocess.

        Returns:
        - str: The preprocessed text.
        """
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        preprocessed_text = ' '.join(filtered_tokens)
        return preprocessed_text

    def _create_faiss_index(self):
        """
        Initializes the FAISS index for storing vector embeddings. The type of index depends on the use case.
        Here, an IndexFlatL2 index is used for simplicity, suitable for L2 distance similarity search.
        """
        self.index = faiss.IndexFlatL2(self.dimension)

    def load_and_encode_texts(self, chunk_dir, doc_id):
        """
        Loads and encodes text chunks from a directory, adding them to the FAISS index for later retrieval.
        Maps the FAISS index positions to the original text chunks for easy lookup.

        Parameters:
        - chunk_dir (str): The directory containing text chunks.
        - doc_id (str): The document identifier, used for organizing text chunks.
        """
        if self.index is None:
            self._create_faiss_index()

        directory = os.path.join(chunk_dir, doc_id)
        for i, filename in enumerate(os.listdir(directory)):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                preprocessed_text = self.preprocess_text(text)
                embedding = self.model.encode([preprocessed_text], convert_to_tensor=True).cpu().detach().numpy()
                if not self.index.is_trained:
                    self.index.train(embedding)
                self.index.add(embedding)
                node_id = len(self.id_to_text)
                self.id_to_text[node_id] = (doc_id, i)

    def update_index(self, chunk_dir):
        """
        Updates the FAISS index with text chunks from all documents in the given directory.

        Parameters:
        - chunk_dir (str): The directory containing folders of text chunks, each folder representing a document.
        """
        for doc_id in os.listdir(chunk_dir):
            path = os.path.join(chunk_dir, doc_id)
            if os.path.isdir(path):
                self.load_and_encode_texts(chunk_dir, doc_id)

    def search(self, query_text, K=5, ef=10, chunk_dir=None):
        """
        Searches the FAISS index for the nearest text chunks to a given query text.

        Parameters:
        - query_text (str): The query text to search for.
        - K (int): The number of nearest neighbors to retrieve.
        - ef (int): Not used in FAISS, included for interface compatibility.
        - chunk_dir (str): The directory containing text chunks for lookup.

        Returns:
        - list: A list of nearest text chunks to the query.
        """
        query_preprocessed = self.preprocess_text(query_text)
        query_vector = self.model.encode([query_preprocessed], convert_to_tensor=True).cpu().detach().numpy()
        D, I = self.index.search(query_vector, K)
        nearest_texts = [self.get_text_from_node(self.id_to_text[n_id], chunk_dir) for n_id in I[0]]
        return nearest_texts

    def get_text_from_node(self, node, chunk_dir):
        """
        Retrieves the text for a given node from the file system using the node's document ID and chunk index.

        Parameters:
        - node (tuple): A tuple containing the document ID and chunk index.
        - chunk_dir (str): The directory containing text chunks.

        Returns:
        - str: The preprocessed text of the chunk.
        """
        with open(f"{os.path.join(chunk_dir, node[0])}/chunk_{node[1]}.txt", "r", encoding="utf-8") as chunk_file:
            return self.preprocess_text(chunk_file.read())

    def reset_index(self):
        """
        Resets the FAISS index and clears any mappings from node IDs to text chunks.
        Optionally recreates the FAISS index.
        """
        self.index = None
        self.id_to_text = {}
        self._create_faiss_index()

    def save_index(self, filepath):
        """
        Saves the current state of the FAISS index and the mapping from node IDs to text chunks to files.

        Parameters:
        - filepath (str): The base filepath for saving the index and mappings.
        """
        faiss.write_index(self.index, filepath + ".faiss")
        with open(filepath + ".pkl", 'wb') as f:
            pickle.dump(self.id_to_text, f)

    def load_index(self, filepath):
        """
        Loads the FAISS index and the mapping from node IDs to text chunks from files.

        Parameters:
        - filepath (str): The base filepath from which to load the index and mappings.
        """
        self.index = faiss.read_index(filepath + ".faiss")
        with open(filepath + ".pkl", 'rb') as f:
            self.id_to_text = pickle.load(f)

# # Create a temporary directory for text chunks
# os.makedirs("temp_data/doc1", exist_ok=True)
# os.makedirs("temp_data/doc2", exist_ok=True)

# # Sample texts
# texts = [
#     ("temp_data/doc1/chunk_0.txt", "The quick brown fox jumps over the lazy dog."),
#     ("temp_data/doc1/chunk_1.txt", "A fast, dark-colored fox leaps above a sleepy canine."),
#     ("temp_data/doc2/chunk_0.txt", "Exploring the efficiency of algorithms in computational linguistics."),
#     ("temp_data/doc2/chunk_1.txt", "Analysis of various search algorithms and their application in NLP."),
# ]

# # Save these texts into files
# for path, content in texts:
#     with open(path, "w") as file:
#         file.write(content)

# # Initialize the FAISSTextRetrieval instance
# faiss_retrieval = FAISSTextRetrieval()

# # Load and encode texts into the FAISS index
# faiss_retrieval.update_index("temp_data")

# # Let's reset the index to simulate loading from disk

# # Perform a search query
# query_text = "algorithm efficiency in NLP"
# nearest_texts = faiss_retrieval.search(query_text, K=2, chunk_dir="temp_data")

# # Display the search results
# print("Search Results:")
# print(nearest_texts)
