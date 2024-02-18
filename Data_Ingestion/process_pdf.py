import fitz
import os
import uuid
import re

# from Data_Ingestion.chunk import Chunk


        
class PDF_Handler():
    def __init__(self):
        return
        
    def chunk_pdf(self, file_path, chunk_path):
        # Extract text and chunk it
        text = self.extract_text_from_pdf(file_path)
        os.makedirs(chunk_path, exist_ok=True)
        for i, chunk in enumerate(list(self.chunk_text(text))):
            with open(f"{chunk_path}/chunk_{i}.txt", "w") as chunk_file:
                chunk_file.write(chunk)
        return
    
    def chunk_text(self, text, chunk_size=250):
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
        
    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts text from a given PDF file.
        """
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    def gen_id(self):
        file_id = str(uuid.uuid4())
        return file_id
    
    def gen_paths(self, UPLOAD_DIR, CHUNK_DIR, file_id):
        file_path = f"{UPLOAD_DIR}/{file_id}.pdf"
        chunk_path = f"{CHUNK_DIR}/{file_id}"
        return file_path, chunk_path
    
    def save_file(self, file_path, file):
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(file.read())
        return