- Post-processing:
    - Merge and re-rank the results to improve retrieval performance.

Step 1: Document Reranking

The first step in our process involves reranking a set of retrieved documents to identify those most relevant to the user's query. This is achieved using the ReRank class, which utilizes the Cohere platform's rerank capability. Given a query and a list of initial document texts, the rerank method in the ReRank class communicates with Cohere's API to evaluate the relevance of each document to the query. This employs a cross-encoder model that allows for more fine-grained comparison between the query and the retrieved documents, and returns the top N documents, ordered by their relevance. This step ensures that the most pertinent information is selected for further processing, significantly improving the efficiency and accuracy of the subsequent question answering phase.

Step 2: Prompt Construction

After reranking, the selected documents are used to construct a comprehensive prompt for the RAG model. This is handled by the Process_Prompt class, which takes the query and the reranked documents as input. The process method first collates the texts of the reranked documents into a knowledge base. Then, it formats this knowledge base along with the original query into a structured prompt. This prompt is specifically designed to provide the RAG model with all necessary context and information, enabling it to generate more accurate and detailed answers. The system prompts and other elements of the message chain is incorporated in the Generation/ Module.