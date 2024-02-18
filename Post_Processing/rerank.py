import cohere  # Ensure cohere is imported at the top

class ReRank():
    """
    This class is designed to rerank a list of document texts based on their relevance to a given query,
    utilizing the Cohere platform's rerank capability.
    """
    
    def __init__(self, api_key="ZsxJgR6bZiQtAIK4gZXni7LmUjMK02WAYiJKyVUA"):
        """
        Initializes the ReRank class with a Cohere API key.

        Parameters:
        - api_key (str): The API key for accessing Cohere's services.
        """
        self.api_key = api_key  # Cohere API key for authentication
        self.co = cohere.Client(api_key)  # Initialize Cohere client with the provided API key

    def rerank(self, query, docs):
        """
        Reranks the provided documents based on their relevance to the given query using Cohere's rerank model.

        Parameters:
        - query (str): The query or question based on which the documents are to be reranked.
        - docs (list of str): A list of document texts to be reranked.

        Returns:
        - list of str: A list of the top N reranked text chunks based on relevance to the query.
        """
        print("start rerank\n")
        # Call Cohere's rerank API with the specified model, query, documents, and the number of top documents to retrieve
        rerank_outputs = self.co.rerank(model="rerank-english-v2.0", query=query, documents=docs, top_n=3)
        
        print("reranked retrievals: ", rerank_outputs)
        # Extract the text of the top N reranked documents
        text_chunks = [result.document['text'] for result in rerank_outputs.results]
        
        return text_chunks
