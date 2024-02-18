Deploy/ contains all scripts and subdirectories needed to run this RAG system.
Clone this repository locally, and navigate to the Deploy/ subdirectory in the command line.

Setting the retrieval method: You can pick between a custom HNSW implementation, and a FAISS implementation (must set a temporary environment variable to do so, as shown below):
For HNSW:
export SEARCH_METHOD=HNSW
For FAISS:
export SEARCH_METHOD=FAISS

Type the above command into the command line

To run the application and setup the server:

run 'uvicorn app:app --reload' in command line (Note: this needs to be run from inside the Deploy/ directory)

uvicorn app:app --reload

In another terminal tab, run the cml.py file. This will allow you to interact with the running server through the command line - instructions will be provided upon startup. Reference the console outputs from the uvicorn application for informative messages as commands are passed in.

A Data/ subdirectory will be created inside of Deploy/. This contains text_chunks/ (which will contain all the active text chunks, organized inside subdirectories labelled by the random doc id generated for each document). It also contains uploaded_files/, which contains the pdf files that have been uploaded thus far.

cml.py instructions:
run the cml.py file

Use /chat <query> to generate a response. Example: /chat what is self attention?
Use /upload <filepath1>:<nickname1> <filepath2>:<nickname2> ... to upload files with nicknames. Example: /upload path/to/file.pdf:nickname
Use /delete <nickname1> <nickname2> ... to delete files by nicknames. Example: /delete nickname
Use /list to list all files. Example: /list
Use /help to see this menu again. Example: /help
Type /quit to exit the program. Example: /quit