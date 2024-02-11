from fastapi import FastAPI, File, UploadFile, HTTPException
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

from Data_Ingestion.process_pdf import *
from Data_Ingestion.chunk import *

@app.post("/upload_files/")
async def upload_files(files: list[UploadFile] = File(...)):
    for file in files:
        file_id = str(uuid.uuid4())
        file_path = f"{UPLOAD_DIR}/{file_id}.pdf"
        chunk_path = f"{CHUNK_DIR}/{file_id}"

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extract text and chunk it
        text = extract_text_from_pdf(file_path)
        os.makedirs(chunk_path, exist_ok=True)
        for i, chunk in enumerate(chunk_text(text)):
            with open(f"{chunk_path}/chunk_{i}.txt", "w") as chunk_file:
                chunk_file.write(chunk)
                
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



### USAGE INSTRUCTIONS ###
'''
run 'uvicorn app:app --reload' in command line
'''