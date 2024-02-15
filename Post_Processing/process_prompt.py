from Post_Processing.rerank import ReRank

class Process_Prompt():
    def __init__(self):
        self.RR = ReRank()
        
    def process(self, query, docs):
        reranked_text_chunks = self.RR.rerank(query=query, docs=docs)
        formatted_prompt = self.construct_prompt(base_query=query, text_chunks=reranked_text_chunks)
        
    def construct_prompt(self, base_query, text_chunks):
        # Assuming 'text_chunks' contains the texts from reranked documents
        knowledge_base = "\n\n".join(text_chunks)
        # Prepare prompt
        prompt = f"Based on the following information:\n{knowledge_base}\n\nAnswer the question: {base_query}"

        return prompt