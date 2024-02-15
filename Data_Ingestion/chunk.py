import re

class Chunk():
    def __init__(self):
        self.a = 1
    
    def chunk_text(text, chunk_size=250):
        """
        Splits text into chunks of a specified size in characters, attempting to respect sentence boundaries.
        """
        # Split the text into sentences
        sentences = re.split(r'(?<=[.!?]) +', text)
        
        chunk = ""
        for sentence in sentences:
            # Check if adding the next sentence would exceed the chunk size
            if len(chunk) + len(sentence) > chunk_size:
                # Yield the current chunk and start a new one
                yield chunk
                chunk = sentence
            else:
                # Add the sentence to the current chunk
                chunk += (' ' + sentence if chunk else sentence)
        
        # Yield any remaining text as the last chunk
        if chunk:
            yield chunk
            
