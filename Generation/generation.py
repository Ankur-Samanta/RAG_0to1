import requests
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

class LLM():
    def __init__(self, api_key="gwGmq0uYH2PWtj3ZnuNpDXeCgtqaXnOf", model="mistral-medium"):
        self.model = model
        self.api_key = api_key
        self.client = MistralClient(api_key=self.api_key)

    def rag_chat(self, prompt):
        return self.extract_content(self.client.chat(model=self.model, messages=self.construct_message_RAG(prompt)))    
    
    def chat(self, prompt):
        return self.extract_content(self.client.chat(model=self.model, messages=self.construct_message_reg(prompt)))    
    
    def intent(self, prompt):
        return self.extract_content(self.client.chat(model=self.model, messages=self.construct_message_intent(prompt)))
        
    def stream(self, prompt):
        return [self.extract_content(completion) for completion in self.client.chat_stream(model=self.model, messages=self.construct_message_RAG(prompt))]
            
    def construct_message_RAG(self, prompt):
        # Sample RAG prompt
        messages = [
            ChatMessage(role="system", content="You are a helpful chatbot assistant who can accurately answer questions, and can reference information from sources provided by the user."),
            ChatMessage(role="user", content=prompt)
        ]
        return messages
    
    def construct_message_reg(self, prompt):
        messages = [
            ChatMessage(role="system", content="You are a helpful chatbot assistant."),
            ChatMessage(role="user", content=prompt)    
        ]
        return messages
    
    def construct_message_intent(self, prompt):
        messages = [
            ChatMessage(role="system", content="You are an intent classifier bot as part of Retrieval Augmented Generation System. Your task is to assess user query intent and categorize the query into one of the following predefined categories: 'generic question', specific question requiring information', and 'generic conversation'"),
            ChatMessage(role="system", content="An example of a 'generic question' is 'what is the weather?' This does not require is to reference any knowledge from a database, but rather a general question that a RAG system likely would not have an answer for'. An example of a 'specific question requiring information' is 'How does self-attention work?' In this case, it is asking a question that requires factual information and generally is better to cite sources for. 'Generic conversation' is any other query that you assess would not need retrieved information to answer"),
            ChatMessage(role="system", content="Make sure to respond ONLY with one of the 3 classifications: 'generic question', 'specific question requiring information', and 'generic conversation'"),
            ChatMessage(role="user", content=prompt)
        ]
        return messages

    def extract_content(self, chat_response):
        return chat_response.choices[0].message.content