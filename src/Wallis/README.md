# Wallis case

## Folder structure
This repository is organized into the following folders:

### `data/`
- Contains raw or intermediate datasets used across the project.
- May include extracted texts, embeddings, metadata files, or external data sources.

### `evaluation/`
- Scripts to generate QA dataset, filetering, and evaulate RAG.
- Includes evaluation metrics, scoring functions, etc.
- Includes notebooks to check and vizualize some results. Includes notebooks of some experiments.

### `original_documents/`
- The original documents, texts of laws after scraping before any transformation or preprocessing.


### `RAGs/`
- Code and components related to Retrieval-Augmented Generation (RAG) systems.
- Includes a python script to launch the RAG and chat with it
- Includes a pkl file containing the default VectorBase (for quicker RAG instantiation) (overlap of 300, chunk_size=[500,1000])

### `training/`
- Training scripts, model checkpoints, and configurations for LoRA finetuning.
- the last FT model is in /training/models/outputs_18-03/checkpoint-54
- the models checkpoints are not pushed to BitBucket (or GitHub)
- To transform a chekpoint to an Ollama model, you must create a modelfile from the base model modelfile, then add the ADAPTER line as :"""
FROM llama3.1:8b\
ADAPTER ./training/outputs_18-03/checkpoint-54\
SYSTEM """You are an AI assistant answering questions about the canton of Valais ..."""\
then you create an ollama model using e.g:
```bash
ollama create ftv1_llama3.1:8b -f Modelfile
``` 

### `transformed_documents/`
- Preprocessed or transformed versions of the `original_documents`. Basically it only add the title of the document (law title) as the first line of the text (when not already there)
- Used for downstream tasks such as embedding generation, RAG ingestion