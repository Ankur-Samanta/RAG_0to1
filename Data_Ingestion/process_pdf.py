import fitz
import os
import uuid
import re

class PDF_Handler():
    """
    A class for handling PDF documents, including extracting text from PDFs and chunking the text into smaller parts.
    """

    def __init__(self):
        """
        Initializes the PDF_Handler class.
        """
        # Currently, there's no initialization code needed.
        pass

    def chunk_pdf(self, file_path, chunk_path):
        """
        Extracts text from a PDF file and chunks it into smaller text files.

        Parameters:
        - file_path (str): The path to the PDF file to be processed.
        - chunk_path (str): The directory path where chunked text files will be saved.
        """
        text = self.extract_text_from_pdf(file_path)
        os.makedirs(chunk_path, exist_ok=True)
        for i, chunk in enumerate(self.chunk_text(text)):
            with open(f"{chunk_path}/chunk_{i}.txt", "w", encoding='utf-8') as chunk_file:
                chunk_file.write(chunk)

    def chunk_text(self, text, chunk_size=250):
        """
        Generator function that splits the given text into chunks of specified size, trying to maintain sentence boundaries.

        Parameters:
        - text (str): The text to be chunked.
        - chunk_size (int): The maximum size of each text chunk in characters.

        Yields:
        - str: Text chunks of up to chunk_size characters.
        """
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunk = ""
        for sentence in sentences:
            if len(chunk) + len(sentence) > chunk_size:
                yield chunk
                chunk = sentence
            else:
                chunk += (' ' + sentence if chunk else sentence)
        if chunk:
            yield chunk

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts all text from a PDF file.

        Parameters:
        - pdf_path (str): The path to the PDF file.

        Returns:
        - str: The extracted text from the PDF.
        """
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        return text

    def gen_id(self):
        """
        Generates a unique identifier for a file.

        Returns:
        - str: A unique identifier string.
        """
        return str(uuid.uuid4())

    def gen_paths(self, UPLOAD_DIR, CHUNK_DIR, file_id):
        """
        Generates file paths for saving a PDF and its chunks based on a unique file ID.

        Parameters:
        - UPLOAD_DIR (str): The directory to save the uploaded PDF.
        - CHUNK_DIR (str): The directory to save the text chunks.
        - file_id (str): The unique file identifier.

        Returns:
        - tuple: A tuple containing the file path and chunk path.
        """
        file_path = f"{UPLOAD_DIR}/{file_id}.pdf"
        chunk_path = f"{CHUNK_DIR}/{file_id}"
        return file_path, chunk_path

    def save_file(self, file_path, file):
        """
        Saves an uploaded file to the specified file path.

        Parameters:
        - file_path (str): The path where the file will be saved.
        - file (File): The file object to be saved.
        """
        with open(file_path, "wb") as buffer:
            buffer.write(file.read())
