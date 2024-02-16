import requests
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

class LLM():
    def __init__(self, api_key="ibe3vQcJcEoA9CyAHc7amv6iijPYzoQT", model="mistral-medium"):
        self.model = model
        self.api_key = api_key
        self.client = MistralClient(api_key=self.api_key)

    def rag_chat(self, prompt):
        return self.extract_content(self.client.chat(model=self.model, messages=self.construct_message_RAG(prompt)))    
    
    def chat(self, prompt):
        print(prompt)
        return self.extract_content(self.client.chat(model=self.model, messages=self.construct_message_reg(prompt)))    
        
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
    
    def extract_content(self, chat_response):
        return chat_response.choices[0].message.content