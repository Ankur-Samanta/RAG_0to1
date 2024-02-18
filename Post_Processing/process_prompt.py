from Post_Processing.rerank import ReRank

class Process_Prompt():
    """
    This class is designed to process and prepare prompts for a Q&A system by incorporating relevant context from reranked text chunks.
    """
    
    def __init__(self):
        """
        Initializes the Process_Prompt class with a ReRank object.
        """
        self.RR = ReRank()  # Instance of a reranking class to improve relevance of text chunks to the query.
        
    def process(self, query, docs):
        """
        Processes the input query and documents to generate a formatted prompt for the Q&A system.

        Parameters:
        - query (str): The user's question or query.
        - docs (list of str): A list of document texts to be considered for answering the query.

        Returns:
        - str: A formatted prompt that combines reranked text chunks with the base query for the Q&A model.
        """
        # Rerank the provided document texts based on their relevance to the query
        reranked_text_chunks = self.RR.rerank(query=query, docs=docs)
        # Construct a comprehensive prompt using the reranked texts and the original query
        formatted_prompt = self.construct_prompt(base_query=query, text_chunks=reranked_text_chunks)
        return formatted_prompt
        
    def construct_prompt(self, base_query, text_chunks):
        """
        Constructs a formatted prompt by integrating reranked text chunks with the base query.

        Parameters:
        - base_query (str): The original user query.
        - text_chunks (list of str): Reranked texts selected to provide context for answering the query.

        Returns:
        - str: A formatted prompt that includes the context and the query for the Q&A model.
        """
        # Concatenate reranked text chunks to form a knowledge base
        knowledge_base = "\n\n".join(text_chunks)
        # Format the prompt to include both the knowledge base and the query for the Q&A system
        prompt = f"Based on the following information:\n{knowledge_base}\n\nAnswer the question: {base_query}"

        return prompt
