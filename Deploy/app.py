from fastapi import FastAPI, File, UploadFile, HTTPException
from sentence_transformers import SentenceTransformer

from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
import os
import uuid


app = FastAPI()

# Directory to store uploaded files and chunks
UPLOAD_DIR = "Data/uploaded_files"
CHUNK_DIR = "Data/text_chunks"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)

# file processing and chunking
from Data_Ingestion.process_pdf import *
from Data_Ingestion.chunk import *
PDF = PDF_Handler()

# indexing for vector search
from Semantic_Search.ANN.hnsw import HNSW
ann = HNSW()

# processing query
from Query_Processing.process_query import analyze_query

# rerank documents and format prompt for rag
from Post_Processing.process_prompt import Process_Prompt
process = Process_Prompt()

# prompt completion generation module
from Generation.generation import LLM
rag = LLM()

@app.post("/upload_files/")
async def upload_files(files: list[UploadFile] = File(...)):
    for file in files:
        file_id = PDF.gen_id()
        file_path, chunk_path = PDF.gen_paths(UPLOAD_DIR, CHUNK_DIR, file_id)

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
            
        # Extract text and chunk it
        PDF.chunk_pdf(file_path, chunk_path)
        ann.update_index_from_doc_dir(chunk_dir=CHUNK_DIR, doc_id=file_id, model=model)
                
    return JSONResponse(status_code=200, content={"message": "Files uploaded and processed successfully."})

@app.delete("/delete_files/")
async def delete_files(file_ids: list[str]):
    for file_id in file_ids:
        file_path = f"{UPLOAD_DIR}/{file_id}.pdf"
        chunk_path = f"{CHUNK_DIR}/{file_id}"
        
        # Delete the PDF file
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            return HTTPException(status_code=404, detail=f"File {file_id} not found.")
        
        # Delete associated chunks
        if os.path.exists(chunk_path):
            for chunk_file in os.listdir(chunk_path):
                os.remove(os.path.join(chunk_path, chunk_file))
            os.rmdir(chunk_path)
        else:
            return HTTPException(status_code=404, detail=f"Chunks for file {file_id} not found.")
            
    return JSONResponse(status_code=200, content={"message": "Files and associated chunks deleted successfully."})

def generate(prompt: str) -> str:
    # Placeholder for the actual implementation
    # This function takes a prompt and returns a generated string
    return "Generated response based on the prompt."

@app.post("/generate_response/")
async def generate_response(prompt: str):
    try:
        # Call the generate function with the user-provided prompt

        # analyze and pre-process prompt
        query = analyze_query(prompt)
        
        # retrieving relevant documents
        query_embedding = model.encode(query)
        nearest_nodes = ann.search_knn(query_embedding, k=5)
        retrieved_documents = [ann.get_text_from_node(node, CHUNK_DIR) for node in nearest_nodes]
        
        # post-process/format prompt and rerank documents
        processed_prompt = process.process(prompt, retrieved_documents)
        
        # generate response
        completion = rag.chat(processed_prompt)

        # Return the generated response to the user
        return JSONResponse(status_code=200, content={"generated_response": completion})
    except Exception as e:
        # Handle potential errors
        return HTTPException(status_code=500, detail=str(e))


model = SentenceTransformer('all-MiniLM-L6-v2')

ann = HNSW.load_index()

### USAGE INSTRUCTIONS ###
'''
run 'uvicorn app:app --reload' in command line
'''