- Data Ingestion:
    - Develop an API endpoint to upload a set of files.
    - Implement a text extraction and chunking algorithm from PDF files. Write down your considerations.

We start by developing a robust API endpoint that serves as the gateway for handling files. This endpoint is designed to handle multiple files concurrently - the relevant API implementation is in Deploy/.

Once a file is uploaded, our focus shifts to the critical process of text extraction and chunking from PDF files. For text extraction, we extract text from each page of the PDF document.The extracted text is then subjected to a chunking algorithm, which splits the text into manageable chunks. This chunking is not arbitrary; it respects sentence boundaries to ensure the coherence and completeness of the text in each chunk. We set a default chunk size, which can be adjusted based on the specific requirements of the subsequent processing stages. This approach facilitates more efficient processing in later stages, such as text analysis or machine learning tasks, by breaking down the text into smaller, more manageable units. This is a naive implementation, however a more informed approach would include semantic chunking (grouping similar chunks of texts together).

The chunked text is then saved into a specified directory, with each chunk as a separate file, making it easily accessible for further processing.

