import string

from Query_Processing.intent import detect_intent, detect_intent_with_model
from Query_Processing.preprocess_query import preprocess_query


def preprocess_documents(documents):
    """Preprocess retrieved documents. Placeholder for actual logic."""
    # This is where you might summarize or chunk documents
    # For simplicity, we'll just return the documents as-is
    return documents

def format_for_rag(query, documents):
    """Format the query and documents for RAG input."""
    formatted_input = f"Answer the following query: {query}\n"
    formatted_input += "using information from the following documents:\n"
    for i, doc in enumerate(documents, start=1):
        formatted_input += f"Document {i}: {doc}\n"
    return formatted_input

def process_query_for_rag(query, documents):
    """Process a query and prepare the input for a RAG model."""
    preprocessed_query = normalize_query(query)
    preprocessed_documents = preprocess_documents(documents)
    rag_prompt = format_for_rag(preprocessed_query, preprocessed_documents)
    return rag_prompt


def process_query(query):
    #intent = detect_intent(query)  
    intent = detect_intent_with_model(query)
    
    if intent == "informational":
        preprocessed_query = preprocess_query(query)
        # Further processing like synonym expansion or spell correction
        # Then, use the processed query for retrieval
        return preprocessed_query
    else:
        return None
