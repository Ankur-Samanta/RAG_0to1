from sentence_transformers import SentenceTransformer
import os
import fitz

from Semantic_Search.ANN.hnsw import *
from Data_Ingestion.chunk import *
from Data_Ingestion.process_pdf import *


chunk_dir = "Data/text_chunks/"
doc_id = "b8e7cf2d-bb28-428a-9b4c-2f924d879659"
model = SentenceTransformer('all-MiniLM-L6-v2')

ann = HNSW.load_index()


query = "LLMs allow for an assumption-neutral and multi-faceted approach for stimulus identification."
query = "What is the input tot he model (the last-token sentence embeddings from GPT2)"
# query = "1). In five train participants, we recorded brain responses to the sentences in the baseline set across two scanning sessions (Figure 6A). Participants were instructed to read attentively and think about the sentence’s meaning. To encourage engagement with the stimuli, prior to the session, participants were informed that they would be asked to perform a short memory task after the session (Methods; fMRI experiments). Sentences were presented one at a time for 2 seconds with a 4 second inter-stimulus interval. Each run contained 50 sentences (5:36 minutes) and sentence order was randomized across participants. The language network was defined functionally in each participant using an extensively validated localizer task (e.g., 2,3; Methods; Definition of ROIs). Although the network consists of five areas (two in the temporal lobe and three in the frontal lobe), we treat it here as a functionally integrated system given i) the similarity among the five regions in their functional response profiles across dozens of experiments (e.g., 21,58,96; see Figure 4B,C and SI 4 for evidence of similar preferences for the baseline set in the current data), ii) high inter-regional correlations during naturalistic cognition paradigms (e.g., 81,76,59,60,83,11). To mitigate the effect of collecting data across multiple scanning sessions and to equalize response units across voxels and participants, the blood-oxygen-level-dependent (BOLD) responses were z-scored session- wise per voxel. BOLD responses from the voxels in the LH language network were averaged within each train participant (Methods; Definition of ROIs) and averaged across participants to yield an average language network response to each of the 1,000 baseline set sentences. Encoding model To develop an encoding model of the language network, we fitted a linear model from the representations of a large language model (LLM) to brain responses (an encoding approach; 174). The brain data that were used to fit the encoding model were the averaged LH language network’s response from the n=5 train participants. To map from LLM representations to brain responses, we made use of a linear mapping model. Note that the term “mapping model” refers to the regression model from LLM representations to brain activity, while the term “encoding model” encompasses both the LLM used to transform a sentence into an embedding representation as well as the mapping model. The mapping model was a L2-regularized (“ridge”) regression model which can be seen as placing a zero-mean Gaussian prior on the regression coefficients 175. Introducing the L2- penalty on the weights results in a closed-form solution to the regression problem, which is similar to the ordinary least-squares regression equation: available under aCC-BY-NC-ND 4.0 International license. (which was not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made bioRxiv preprint doi: https://doi.org/10.1101/2023.04.16.537080; this version posted October 30, 2023. The copyright holder for this preprint 22 Where 𝑋 is a matrix of regressors (n stimuli by d regressors). The regressors are unit activations from the sentence representations derived by exposing an LLM to the same stimuli as the"



query_embedding = model.encode(query)
nearest_nodes = ann.search_knn(query_embedding, k=5)
retrieved_text = [ann.get_text_from_node(node, chunk_dir) for node in nearest_nodes]

for text in retrieved_text:
    print(text)
    print("\n\n")