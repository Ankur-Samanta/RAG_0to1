import requests
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

class LLM():
    """
    A class to interact with a Language Model (LLM) for various tasks including chat, intent classification, and content extraction.
    
    Attributes:
        model (str): The model name to be used for generating responses.
        api_key (str): API key for authentication to access the model.
        client (MistralClient): Client to interact with the Mistral API.
    """
    
    def __init__(self, api_key="gwGmq0uYH2PWtj3ZnuNpDXeCgtqaXnOf", model="mistral-medium"):
        """
        Initializes the LLM class with a specific model and API key.
        
        Parameters:
            api_key (str): API key for Mistral client authentication.
            model (str): Name of the model to use for response generation.
        """
        self.model = model
        self.api_key = api_key
        self.client = MistralClient(api_key=self.api_key)

    def rag_chat(self, prompt):
        """
        Generates a chat response for RAG-based queries.
        
        Parameters:
            prompt (str): User's prompt or question.
            
        Returns:
            str: The content extracted from the chat response.
        """
        return self.extract_content(self.client.chat(model=self.model, messages=self.construct_message_RAG(prompt)))

    def chat(self, prompt):
        """
        Generates a standard chat response.
        
        Parameters:
            prompt (str): User's prompt or question.
            
        Returns:
            str: The content extracted from the chat response.
        """
        return self.extract_content(self.client.chat(model=self.model, messages=self.construct_message_reg(prompt)))

    def intent(self, prompt):
        """
        Determines the intent of a given prompt.
        
        Parameters:
            prompt (str): User's prompt or question.
            
        Returns:
            str: The content extracted from the chat response, indicating the intent classification.
        """
        return self.extract_content(self.client.chat(model=self.model, messages=self.construct_message_intent(prompt)))

    def stream(self, prompt):
        """
        Streams responses for a given prompt, useful for continuous interactions.
        
        Parameters:
            prompt (str): User's prompt or question.
            
        Returns:
            list[str]: A list of content extracted from streamed chat responses.
        """
        return [self.extract_content(completion) for completion in self.client.chat_stream(model=self.model, messages=self.construct_message_RAG(prompt))]

    def construct_message_RAG(self, prompt):
        """
        Constructs a message tailored for RAG-based interactions.
        
        Parameters:
            prompt (str): User's prompt or question.
            
        Returns:
            list[ChatMessage]: A list of ChatMessage objects for RAG interactions.
        """
        # Defines the role and content of each message in the chat sequence for RAG.
        messages = [
            ChatMessage(role="system", content="You are a helpful chatbot assistant who can accurately answer questions, and can reference information from sources provided by the user."),
            ChatMessage(role="user", content=prompt)
        ]
        return messages

    def construct_message_reg(self, prompt):
        """
        Constructs a standard message for regular chat interactions.
        
        Parameters:
            prompt (str): User's prompt or question.
            
        Returns:
            list[ChatMessage]: A list of ChatMessage objects for regular interactions.
        """
        messages = [
            ChatMessage(role="system", content="You are a helpful chatbot assistant."),
            ChatMessage(role="user", content=prompt)
        ]
        return messages

    def construct_message_intent(self, prompt):
        """
        Constructs a message for intent classification tasks.
        
        Parameters:
            prompt (str): User's prompt or question.
            
        Returns:
            list[ChatMessage]: A list of ChatMessage objects for intent classification.
        """
        messages = [
            ChatMessage(role="system", content="You are an intent classifier bot as part of Retrieval Augmented Generation System. Your task is to assess user query intent and categorize the query into one of the following predefined categories: 'generic question', 'specific question requiring information', and 'generic conversation'"),
            ChatMessage(role="system", content="An example of a 'generic question' is 'what is the weather?' This does not require us to reference any knowledge from a database, but rather a general question that a RAG system likely would not have an answer for. An example of a 'specific question requiring information' is 'How does self-attention work?' In this case, it is asking a question that requires factual information and generally is better to cite sources for. 'Generic conversation' is any other query that you assess would not need retrieved information to answer."),
            ChatMessage(role="system", content="Make sure to respond ONLY with one of the 3 classifications: 'generic question', 'specific question requiring information', and 'generic conversation'"),
            ChatMessage(role="user", content=prompt)
        ]
        return messages

    def extract_content(self, chat_response):
        """
        Extracts the message content from a chat response.
        
        Parameters:
            chat_response (ChatResponse): The chat response object from Mistral.
            
        Returns:
            str: The content of the first choice in the chat response.
        """
        return chat_response.choices[0].message.content
