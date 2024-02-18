import cohere

class ReRank():
    def __init__(self, api_key="ZsxJgR6bZiQtAIK4gZXni7LmUjMK02WAYiJKyVUA"):
        self.api_key = api_key
        self.co = cohere.Client(api_key)

    def rerank(self, query, docs):
        print("start rerank\n")
        rerank_outputs = self.co.rerank(model="rerank-english-v2.0", query=query, documents=docs, top_n=3)
        # Extracting text chunks
        print(rerank_outputs)
        text_chunks = [result.document['text'] for result in rerank_outputs]
        print("c\n")
        return text_chunks