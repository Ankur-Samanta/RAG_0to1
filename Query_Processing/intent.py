from transformers import pipeline

def detect_intent(query):
    informational_keywords = ["what is", "how to", "tell me about"]
    for keyword in informational_keywords:
        if keyword in query.lower():
            return "informational"
    return "non-informational"


def detect_intent_with_model(query):
    classifier = pipeline("zero-shot-classification")
    candidate_labels = ["informational", "greeting", "question", "command"]
    result = classifier(query, candidate_labels)
    return result["labels"][0]

