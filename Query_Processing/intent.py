from transformers import pipeline
from Generation.generation import LLM

class Intent():
    """
    A class for classifying the intent of a query using a two-step approach:
    first utilizing a Large Language Model (LLM) and then refining the classification
    with a zero-shot classification model.
    """
    
    def __init__(self):
        """
        Initializes the Intent class by setting up the required models for intent
        classification.
        """
        self.llm = LLM()  # Placeholder for a Large Language Model instance.
        self.classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")  # Zero-shot classification pipeline.
        
        # Predefined candidate labels for classification.
        self.candidate_labels = ['generic question', 'specific question requiring information', 'generic conversation']

    def intent_classifier(self, query):
        """
        Classifies the intent of a given query by first processing it with an LLM and then
        using a zero-shot classifier for fine-grained intent detection.

        Parameters:
        - query (str): The user's query.

        Returns:
        - str: The classified intent of the query.
        """
        # Use LLM to determine intent and nature of the query.
        llm_intent = self.llm.intent(query)
        # Pass LLM output into zero shot classifier for final label assignment.
        classified_intent = self.classifier(llm_intent, self.candidate_labels)['labels'][0]
        return classified_intent
        
    def use_rag(self, query):
        """
        Determines whether a Retrieval-Augmented Generation (RAG) model should be used
        based on the classified intent of the query.

        Parameters:
        - query (str): The user's query.

        Returns:
        - bool: True if a RAG model should be used (for specific information queries),
                False otherwise.
        """
        classified_intent = self.intent_classifier(query)
        # Use RAG for specific questions requiring information retrieval.
        if classified_intent == 'specific question requiring information':
            return True
        else: 
            return False

            
# intent = Intent()

# sequence_to_classify = "how can you test whether current models of the human language network are capable of driving and suppressing brain responses?"

# print(intent.use_rag(sequence_to_classify))