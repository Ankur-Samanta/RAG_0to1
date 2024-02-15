import os

from Data_Ingestion.chunk import *
from Data_Ingestion.process_pdf import *

filepath = "Data/uploaded_files/b8e7cf2d-bb28-428a-9b4c-2f924d879659.pdf"
UPLOAD_DIR = "Data/uploaded_files"
CHUNK_DIR = "Data/text_chunks"

file_id = "b8e7cf2d-bb28-428a-9b4c-2f924d879659"
file_path = f"{UPLOAD_DIR}/{file_id}.pdf"
chunk_path = f"{CHUNK_DIR}/{file_id}"

PDF = PDF_Handler()
# Extract text and chunk it
PDF.chunk_pdf(file_path, chunk_path)