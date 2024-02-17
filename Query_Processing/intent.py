from transformers import pipeline
from Generation.generation import LLM

class Intent():
    def __init__(self):
        self.llm = LLM()
        self.classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
        
        self.candidate_labels = ['generic question', 'specific question requiring information', 'generic conversation']

    def intent_classifier(self, query):
        # Use LLM to determine intent and nature of the query
        llm_intent = self.llm.intent(query)
        # Pass LLM output into zero shot classifier for final label assignment
        classified_intent = self.classifier(llm_intent, self.candidate_labels)['labels'][0]
        return classified_intent
        
    def use_rag(self, query):
        classified_intent = self.intent_classifier(query)
        if classified_intent == 'specific question requiring information':
            return True
        else: 
            return False
            
        
# intent = Intent()

# sequence_to_classify = "how can you test whether current models of the human language network are capable of driving and suppressing brain responses?"

# print(intent.use_rag(sequence_to_classify))