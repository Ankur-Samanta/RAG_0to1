from sentence_transformers import SentenceTransformer
import os
import fitz

from Semantic_Search.ANN.hnsw import *
from Data_Ingestion.chunk import *
from Data_Ingestion.process_pdf import *


chunk_dir = "Data/text_chunks/"
doc_id = "b8e7cf2d-bb28-428a-9b4c-2f924d879659"
model = SentenceTransformer('all-MiniLM-L6-v2')


ann = HNSW(initial_dataset_size=655)
ann.update_index_from_doc_dir(chunk_dir=chunk_dir, doc_id=doc_id, model=model)

ann.save_index("Data/index/ann_index.pkl")

