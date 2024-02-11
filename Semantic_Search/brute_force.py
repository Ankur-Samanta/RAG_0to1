

# Placeholder for embedding generation
def generate_embedding(text):
    # Implement embedding generation
    # For demonstration, return a random vector
    import numpy as np
    return np.random.rand(300)  # Assuming 300-dimensional embeddings

documents = []
# Assuming `documents` is a list of texts
document_embeddings = {doc_id: generate_embedding(text) for doc_id, text in documents.items()}


import numpy as np

def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)

def batch_cosine_similarity(query_vec, doc_vecs):
    # Normalize the query vector to unit length
    query_vec_norm = query_vec / np.linalg.norm(query_vec)
    
    # Normalize document vectors to unit length
    doc_vecs_norm = doc_vecs / np.linalg.norm(doc_vecs, axis=1)[:, np.newaxis]
    
    # Compute cosine similarity as dot products
    similarities = np.dot(doc_vecs_norm, query_vec_norm)
    
    return similarities

def vector_search(query_embedding, document_embeddings, top_k=10):
    similarities = {}
    for doc_id, doc_embedding in document_embeddings.items():
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities[doc_id] = similarity
    
    # Sort documents by similarity
    sorted_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Return top_k most similar documents
    return sorted_docs[:top_k]

def efficient_vector_search(query_embedding, document_embeddings, top_k=10):
    # Calculate similarities for all documents in one batch
    similarities = batch_cosine_similarity(query_embedding, document_embeddings)
    
    # Get the top_k most similar document indices
    top_indices = np.argsort(-similarities)[:top_k]
    
    # Retrieve the top_k similarities
    top_similarities = similarities[top_indices]
    
    return top_indices, top_similarities

#BASE
query = "Example search query"
query_embedding = generate_embedding(query)
top_documents = vector_search(query_embedding, document_embeddings, top_k=5)

# Assuming you have a way to fetch documents by their IDs
for doc_id, similarity in top_documents:
    print(f"Doc ID: {doc_id}, Similarity: {similarity}")


#EFFICIENT

# Example embeddings for 1000 documents, each with an embedding size of 300
num_documents = 1000
embedding_size = 300
document_embeddings = np.random.rand(num_documents, embedding_size)

# Generate a query embedding
query = "Example search query"
query_embedding = generate_embedding(query)  # Your actual function to generate an embedding

# Perform the search
top_indices, top_similarities = efficient_vector_search(query_embedding, document_embeddings, top_k=5)

# Output results
print("Top Documents and their Similarities:")
for idx, sim in zip(top_indices, top_similarities):
    print(f"Document Index: {idx}, Similarity: {sim}")
