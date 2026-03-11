# CIT case
You can check the [Medium article](https://medium.com/author-page) about how we finetuned the LLMs for the Corporate IT Knowledge base.

## How to use (focus on CIT CASE)
You can launch the scraping launching from the root folder
```bash
make scrape
``` 
It will write or overwrite the documents of the KB in ./src/CIT/documents/run2/confluence_json. To set up different a different path you can modify the parameters in ./src/CIT/scraping/settings.toml

Then you can launch the RAG chatbot in a terminal (with the default arguments) with:
```bash
make run_rag
``` 
If you need to adapt the parameters (chunks size, chunk overlap, top_k, threshold, model etc.) you can go to ./src/CIT/RAGs/launch_RAG.sh and set up the desired RAG.

If you want to launch the user interface:\
DEV (with hand on the RAG parameters and ongoing trials): The following command launch the streamlit app in ./src/CIT/UI/streamlit_app.py
```bash
make launch_ui_dev
``` 
PROD (no hands on parameters): launch the streamlit app in ./src/CIT/UI/streamlit_app_prod.py
```bash
make launch_ui_prod
``` 

the defaults parameters are in the python script:
```python
    DIRECTORY = "./src/CIT/documents/run3/confluence_json" # path to the documents directory, where the documents are stored (it is where the scraping script writes the documents, it should contain a folder called "mappings" containing the mappings between ids, urls, etc. as created by the scraping script)
    CHUNK_SIZE=[1500] # size of the chunks to split the documents into, to avoid long context windows and to have a better retrieval performance
    CHUNK_OVERLAP=300 # overlap between chunks, to avoid losing information at the end of a chunk
    EMBEDDING_MODEL = "./src/CIT/RAGs/models/all-MiniLM-L6-v2"  # Path to the embedding model
    NB_CHUNKS = -1 # -1 means all chunks, used to debug, you should set it to -1

  
    MODEL_NAME = "ft_wo0" # Name of the model to use for the RAG
    NUM_PREDICT_TOKENS = 1000 # max number of tokens per answer
    TOP_K = 6 # number of top k documents to retrieve
    THRESHOLD = 0.1 # threshold for the similarity score to consider a document relevant, after top_k retrieval
    KEEP_IN_MIND_LAST_N_MESSAGES = 6 # number of last messages to keep in mind for the RAG
    ADD_EXTERNAL_LINKS_DOCS=False # if True, the RAG will add the links to the documents in the answer, we found it not useful for the CIT case
    ALWAYS_DO_RETRIEVAL = False # if True, the RAG will always do the retrieval, even if the question is as simple as "Hello, who are you?"
```

- To run it somewhere else you must have the embedding model from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 downloaded and put in the path specified by `EMBEDDING_MODEL` (default is `./src/CIT/RAGs/models/all-MiniLM-L6-v2`).
Same for the reranking model https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2 




## Folder structure
This repository is organized into the following folders:

### documents/
- Contains the scraped documents, with or without deleting some of the documents. It also contains the mappings between ids, urls etc.
- This folder is not commited. You can create these documents, running the main script from the scraping folder

### evaluation/
#### /QA_generation/
- QA_generation: Scripts to generate QA dataset: automatic answers generation, questions and answers rephrasing, matching rephrased questions and answers
- Includes the data with intermediate results (e.g manual questions creation, snythetic answers etc.). These intermediate results are not commited<br>
The idea is:
     1) Create manually some questions under a jsonl file. Each question must contain a "question", a list of "urls"
     2) Ask an llm to directly answer these questions (answers_questions.py)
     3) Manually review and modify these answers to have a final set of clean data
     4) Rephrase the questions and the answers with rephrase_questions.py and rephrase_answers.py
     5) Match the newly rephrased questions and answers to cross all the generated data and augment the dataset
     
#### Python functions and shell scripts
- the latest evaluation pipeline is in the 'training' folder, in the script `perform_cross_validation_evaluation.py`
- it will compute the answers and the scores for each fold test set of the cross-validation dataset
- python scripts to compute evaluation metrics: retrieval performance and URL based RAG metric
- viz: notebooks to view retrieval performance with different parameters, check visualizations about the results etc.
- scripts folder: shell script to launch the evaluation python script (clearer to set the different parameters), as well as the QA generation described above

#### /results/balanced_split
- not committed
- answers: answers of the RAGs on the test set
- precision_recall: scores of the RAGs using our designed URLs metric
- balanced_split: the results of the different models after balancing the dataset


### RAGs/
- python script (and its correponding shell script) to launch a RAG (different hyperparameters to set)
You can also simply use 
```bash
make run_rag
``` 
at the root of the project, with the desired arguments. It will launch the RAG in the terminal with the default parameters.
- the latest RAG is in the utils.py, not a script to launch, it is preferable to launch the UI directly
- For evaluation, we use the RAG_CIT.py script, which was a previous version of the RAG, to remain consistent across the different evaluations. It is not used anymore in production (though it is very similar to the utils.py RAG). This is also the RAG used to evaluate the baselines, putting model name as "random_urls_model_baseline", and launching the base_cv_eval.py script to compute the answers and the scores on the cross-validation dataset.

### scraping/
- scraping scripts and settings to get the documents from the Confluence space of the Coporate IT Knowledge base
- launch main.py to scrape the Confluence (arguments in seeting.toml)
You can use:\
```bash
make scrape
``` 


### training/
- Training scripts, model checkpoints, and configurations for LoRA finetuning.
To launch the training you can find an example of shell script at src/CIT/training/scripts/llama8b.sh
- the best FT model (the one used in production) is in models/cv/ft_28.5_with_at_least_1_url/model_fold_3
- the models checkpoints are not pushed to BitBucket (or GitHub)
- To transform a chekpoint to an Ollama model, you must create a modelfile from the base model modelfile, then add the ADAPTER line as :"""
FROM llama3.1:8b\
ADAPTER models/cv/ft_28.5_with_at_least_1_url/model_fold_3
SYSTEM """You are an AI assistant bla bla bla ..."""\
then you create an ollama model using e.g:

```bash
ollama create finetunedv1_llama3.1:8b -f Modelfile
``` 

- the answers generated by the different RAG on the cross-validation folds are in models/cv/data/test_answers (not committed)
- the finals scores of the RAGs on the test set are in models/cv/scores (not committed)
- the whole set of created Q&A to train and test the models is in data/split_balance_zeros/urls_formatted/run_22.5/new_retrieved_context/all_data_transformed.jsonl