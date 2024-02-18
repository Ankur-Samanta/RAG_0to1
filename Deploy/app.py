from fastapi import FastAPI, File, UploadFile, HTTPException
from sentence_transformers import SentenceTransformer
from typing import List
from pydantic import BaseModel

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
PDF = PDF_Handler()

# file mapping (uuid id to pdf name and nickname)
files_mapping = {}

# indexing for vector search
from Semantic_Search.HNSW.hnsw_retr import HNSWTextRetrieval
from Semantic_Search.FAISS.faiss_retr import FAISSTextRetrieval

search_method = os.getenv('SEARCH_METHOD', 'FAISS').upper()

if search_method == 'HNSW':
    print("using HNSW for retrieval")
    ann = HNSWTextRetrieval(M=10, Mmax=15, efConstruction=200, mL=1.0) # NOTE: these parameters have not been optimized, so retrieval results will likely be suboptimal
elif search_method == 'FAISS':
    print("using FAISS for retrieval")
    ann = FAISSTextRetrieval()  # Out-of-the-box FAISS ANN implementation as benchmark
else:
    raise ValueError("Unsupported SEARCH_METHOD environment variable value")

# processing query
from Query_Processing.preprocess_query import preprocess_query
from Query_Processing.intent import Intent
intent = Intent()

# rerank documents and format prompt for rag
from Post_Processing.process_prompt import Process_Prompt
process = Process_Prompt()

# prompt completion generation module
from Generation.generation import LLM
rag = LLM()

def update_files_mapping_from_directory(upload_dir):
    """
    Scans the upload directory for PDF files and updates the file mapping.
    """
    for filename in os.listdir(upload_dir):
        if filename.endswith(".pdf"):
            # Extract file ID from the filename (assuming file ID + '.pdf' is the filename)
            file_id = filename[:-4]  # Removes the '.pdf' extension to get the file ID

            # If using a more complex scheme to generate file_id, adjust the above accordingly

            # Check if the file ID already exists in the mapping
            # If not, update the mapping with default or generated values
            if file_id not in files_mapping:
                # Here, you might want to generate a nickname or use some default value
                # For simplicity, we're using the filename without extension as nickname
                nickname = filename[:-4]  # Or any logic to generate a nickname
                
                # Update the mapping with a new entry
                files_mapping[file_id] = {
                    "nickname": nickname,
                    "full_file_name": filename
                }
                print(nickname, "loaded")

@app.on_event("startup")
async def load_index():
    """
    Loads or initializes the document index at application startup.
    """
    global ann
    filename = 'path/to/your/index/file'
    # Attempt to load the index; if it doesn't exist, initialize it
    if os.path.isfile(filename):
        ann = HNSWTextRetrieval(M=10, Mmax=15, efConstruction=200, mL=1.0).load_index(filename=filename)
    else:
        # Initialize the index if the file doesn't exist
        print("loading index")
        ann.update_index(CHUNK_DIR)
        print("index loaded")
        
        # update files list
        update_files_mapping_from_directory(UPLOAD_DIR)

@app.post("/upload_files/")
async def upload_files(files: List[UploadFile] = File(...), nicknames: List[str] = File(...)):
    """
    Endpoint for uploading files. Each file must have a corresponding nickname.
    """
    if len(files) != len(nicknames):
        raise HTTPException(status_code=400, detail="Each file must have a corresponding nickname.")

    responses = []
    for file, nickname in zip(files, nicknames):
        original_name = file.filename
        file_id = PDF.gen_id()
        file_path, chunk_path = PDF.gen_paths(UPLOAD_DIR, CHUNK_DIR, file_id)

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Extract text and chunk it, update the search index, etc.
        PDF.chunk_pdf(file_path, chunk_path)
        ann.load_and_encode_texts(chunk_dir=CHUNK_DIR, doc_id=file_id)
        
        # Store the file mapping
        files_mapping[file_id] = {
            "nickname": nickname,
            "full_file_name": original_name  # Original file name as uploaded by the user
        }

        responses.append({"file_id": file_id, "nickname": nickname, "original_name": original_name})

    print("index updated")
                
    return JSONResponse(status_code=200, content={"message": "Files uploaded and processed successfully.", "files": responses})

@app.get("/list_files/")
async def list_files():
    """
    Endpoint to list all files that have been uploaded and processed.
    """
    files_list = []
    for file_id, file_info in files_mapping.items():
        files_list.append({
            "nickname": file_info["nickname"],
            "original_name": file_info["full_file_name"]
        })
    
    return JSONResponse(content={"files": files_list})



class DeleteRequest(BaseModel):
    '''
    class definition for the input to delete_files
    '''
    ids: List[str]

def get_nicknames():
    '''
    Function to retrieve all the nicknames of files currently stored.
    '''
    nicknames = []    
    for file_id, file_info in files_mapping.items():
        nicknames.append(file_info["nickname"])
    return nicknames

@app.delete("/delete_files/")
async def delete_files(delete_request: DeleteRequest):
    """
    Endpoint to delete files based on a list of nicknames or IDs.
    """
    nicknames = delete_request.ids
    if nicknames[0] == "all":
        nicknames = get_nicknames()
    for nickname in nicknames:

        # Find file_id by nickname
        file_id = None
        for fid, info in files_mapping.items():
            if info["nickname"] == nickname:
                file_id = fid
                break
        
        # If the file_id was not found, return an error
        if not file_id:
            return HTTPException(status_code=404, detail=f"File with nickname '{nickname}' not found.")
        
        file_path = f"{UPLOAD_DIR}/{file_id}.pdf"
        chunk_path = f"{CHUNK_DIR}/{file_id}"
        
        # Delete the PDF file
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            # If the file doesn't exist, it's not necessarily an error since the goal is to ensure it's deleted
            print(f"File {file_id} not found. Skipping.")
        
        # Delete associated chunks
        if os.path.exists(chunk_path):
            for chunk_file in os.listdir(chunk_path):
                os.remove(os.path.join(chunk_path, chunk_file))
            os.rmdir(chunk_path)
        else:
            # If the chunks don't exist, it's not necessarily an error since the goal is to ensure they're deleted
            print(f"Chunks for file {file_id} not found. Skipping.")

        # Remove the entry from the files_mapping after deletion
        del files_mapping[file_id]

        # Reset and update the index if necessary
        # It might be more efficient to do this outside the loop if multiple files are being deleted frequently
        ann.reset_index()
        ann.update_index(CHUNK_DIR)

    return JSONResponse(status_code=200, content={"message": "Files and associated chunks deleted successfully."})

@app.post("/generate_response/")
async def generate_response(prompt: str):
    """
    Endpoint to generate a response to a given prompt, using RAG if appropriate. Will print descriptive messages (i.e whether the query passed in required rag or not, or what chunks were retrieved) to the server console
    """
    try:
        # analyze and pre-process prompt
        query = preprocess_query(prompt)
        use_rag = intent.use_rag(query)
        if use_rag:
            print("use rag")
            # retrieving relevant documents
            retrieved_documents = ann.search(query, K=10, chunk_dir=CHUNK_DIR)
            # post-process/format prompt and rerank documents
            print("retrieved_documents: ", retrieved_documents)
            processed_prompt = process.process(query, retrieved_documents)
            completion = rag.rag_chat(processed_prompt)
        else:
            print("no rag")
            completion = rag.chat(query)
        # generate response

        # Return the generated response to the user
        return JSONResponse(status_code=200, content={"generated_response": completion})
    except Exception as e:
        # Handle potential errors
        return HTTPException(status_code=500, detail=str(e))


### USAGE INSTRUCTIONS ###
'''
Navigate to the Deploy/ subdirectory.

Setting the retrieval method: You can pick between a custom HNSW implementation, and a FAISS implementation
For HNSW:
export SEARCH_METHOD=HNSW
For FAISS:
export SEARCH_METHOD=FAISS

type the above command into the command line in the directory you are running the application

To run the application and setup the server:

run 'uvicorn app:app --reload' in command line

uvicorn app:app --reload

'''