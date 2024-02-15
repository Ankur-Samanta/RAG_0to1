import requests
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from Generation.generation import LLM


rerank_outputs = {
    "results": [
        {
            "document": { "text": "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district. The President of the USA and many major national government offices are in the territory. This makes it the political center of the United States of America." },
            "index": 3,
            "relevance_score": 0.9871293
        },
        {
            "document": { "text": "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan." },
            "index": 1,
            "relevance_score": 0.29961726
        },
        {
            "document": { "text": "Carson City is the capital city of the American state of Nevada. At the 2010 United States Census, Carson City had a population of 55,274." },
            "index": 0,
            "relevance_score": 0.08977329
        },
        {
            "document": { "text": "Charlotte Amalie is the capital and largest city of the United States Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas." },
            "index": 2,
            "relevance_score": 0.041462272
        }
    ]
}

# Extracting text chunks
text_chunks = [result['document']['text'] for result in rerank_outputs['results']]

# Your user's base query
base_query = "What is the capital of the United States?"

# Assuming 'text_chunks' contains the texts from your reranked documents
knowledge_base = "\n\n".join(text_chunks)

# Prepare your prompt
prompt = f"Based on the following information:\n{knowledge_base}\n\nAnswer the question: {base_query}"
    
rag = LLM()
completion = rag.chat(prompt)
print(completion)