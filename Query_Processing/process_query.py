import string

from Query_Processing.intent import detect_intent, detect_intent_with_model
from Query_Processing.preprocess_query import preprocess_query


def analyze_query(query):
    #intent = detect_intent(query)  
    query = preprocess_query(query)
    intent = detect_intent_with_model(query)
    
    if intent == "informational" or "question":
        # Further processing like synonym expansion or spell correction
        # Then, use the processed query for retrieval
        return query, True
    else:
        return query, False
