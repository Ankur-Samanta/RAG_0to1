- Query Processing, given a user query:
    - Detect the intent of the query to assess if triggering a knowledge base search is necessary. For example, “hello” should not trigger a search.
    - Transform the query to improve the retrieving for RAG.

Intent Determination
The first stage involves understanding the nature and intent of the query through a two-step process. Initially, we utilize a large language model (LLM) to analyze the query's context and subtleties, providing a preliminary categorization of the query's intent. This categorization helps in understanding whether the query is a general question, a specific information request, or part of a broader conversation. Subsequently, the output from the LLM is further refined using a zero-shot classification model, which serves as a layer of redundancy to ensure that the output is indeed 1 of 3 categories: 'generic question', 'specific question requiring information', and 'generic conversation'. 

Query Preprocessing
Upon determining a query's intent as requiring factual information, the system progresses to the query preprocessing stage. This stage is designed to optimize the query for the RAG model, ensuring that it is clean, clear, and devoid of ambiguities that could impair the retrieval process. The preprocessing involves several steps:

Normalization: Converts the query to lowercase and removes any punctuation, standardizing the format for consistency.
Spelling Correction: Employs a spell checker to correct any spelling mistakes in the query, reducing the risk of misinterpretations or retrieval errors.
Synonym Expansion (Optional): For key terms within the query, we consider expanding synonyms to broaden the search criteria, although this step is applied selectively based on the context and necessity. It aims to capture various ways of expressing the same information need, enhancing the retrieval model's ability to find relevant documents.

Constructing the prompt for best RAG performance is done in the Post_Processing/ module, as it takes the reranked documents as input.