This Retrieval Augmented Generation (RAG) system is designed to enhance response generation by incorporating external knowledge sources. At its core, the system integrates a vector search mechanism with a language model to fetch relevant information and generate informed responses. The process begins with data ingestion, where documents are uploaded through an API, extracted from PDFs, and chunked into manageable text segments. These segments are then encoded into vector embeddings using a Sentence Transformer model and stored in a vector database using either a custom HNSW (Hierarchical Navigable Small World) implementation or FAISS (Facebook AI Similarity Search) for efficient similarity search.

When a query is received, it undergoes preprocessing to enhance its quality, including normalization, spelling correction, and optionally synonym expansion. The system then determines the query's intent to decide if external knowledge retrieval is necessary. If so, the vector search retrieves the top N relevant text chunks based on the query's embedding. The retrieved chunks are reranked to prioritize the most relevant information, which, along with the original query, forms a comprehensive prompt for the language model.

The language model, powered by a Mistral-based chat interface, generates a response by considering both the query and the context provided by the selected text chunks. This approach ensures that the generated responses are not only relevant but also grounded in the information extracted from the uploaded documents.

Additionally, the system offers functionalities for managing the document repository, including file upload, deletion, and listing, through a FastAPI framework. The interactive script allows for command-line interaction with the API, facilitating operations like file management and response generation.

Flow:
Deploy -> Data_Ingestion -> Query_Processing -> Semantic_Search -> Post_Processing -> Generation -> Deploy

To use this system, proceed to Deploy/ and refer to its readme.

Copyright (2024) - Ankur Samanta - All Rights Reserved
