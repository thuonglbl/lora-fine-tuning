#!/bin/bash

# ------------------------------
# Configurable parameters
# ------------------------------
DIRECTORY="./src/CIT/scraping/JIRA/jira_documents"
CHUNK_SIZE="[1500]"
CHUNK_OVERLAP=300
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
NB_CHUNKS=-1
TOP_K=6
THRESHOLD=0.1
KEEP_IN_MIND_LAST_N_MESSAGES=8
MODEL_NAME="llama3.1:8b"
NUM_PREDICT_TOKENS=1000
ALWAYS_DO_RETRIEVAL=false
VERBOSE=true


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------
# Run the Python script
# ------------------------------
python "$SCRIPT_DIR/RAG_JIRA.py" \
  --directory "$DIRECTORY" \
  --chunk_size "$CHUNK_SIZE" \
  --chunk_overlap "$CHUNK_OVERLAP" \
  --embedding_model "$EMBEDDING_MODEL" \
  --nb_chunks "$NB_CHUNKS" \
  --top_k "$TOP_K" \
  --threshold "$THRESHOLD" \
  --keep_in_mind_last_n_messages "$KEEP_IN_MIND_LAST_N_MESSAGES" \
  --model_name "$MODEL_NAME" \
  --num_predict_tokens "$NUM_PREDICT_TOKENS" \
  --always_do_retrieval "$ALWAYS_DO_RETRIEVAL" \
  --verbose "$VERBOSE"