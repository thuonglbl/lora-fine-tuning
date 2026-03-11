#script to evaluate a base model on a cross-validation dataset
#it will compute the answers and the scores for each fold of the cross-validation dataset

import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from CIT.evaluation.utils import evaluate_rag_model, load_jsonl, save_jsonl, str_to_bool
from CIT.RAGs.RAG_CIT import VectorBase

parser= ArgumentParser()

parser.add_argument(
    "--splits_folder",
    type=str,
    default="./src/CIT/training/models/cv/data/23.5_2epochs",
    help="output directory for the splits data",
)

parser.add_argument(
    "--model_name",
    type=str,
    default="llama3.3:70b",
    help="name of the model to use",
)

parser.add_argument(
    "--test_answers_folder",
    type=str,
    default="./src/CIT/training/models/cv/data/test_answers/ft_23.5_2epochs/base_70b",
    help="folder where to store answers on the different test sets",
)

parser.add_argument(
    "--scores_folder",
    type=str,
    default="./src/CIT/training/models/cv/scores/ft_23.5_2epochs/base_70b",
    help="output directory for the scores",
)

parser.add_argument(
    "--directory",
    type=str,
    default="../documents/confluence_json_without_root_with_titles",
    help="Directory containing the documents to be indexed",
)
parser.add_argument(
    "--chunk_size",
    default=[1500],
    help="Size of the chunks to split the documents into, can be a list if you want to put multiple chunk sizes",
)
parser.add_argument(
    "--chunk_overlap", type=int, default=300, help="Overlap between chunks"
)
parser.add_argument(
    "--embedding_model",
    type=str,
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="Huggingface model to use for embeddings",
)
parser.add_argument(
    "--nb_chunks", type=int, default=-1, help="Number of chunks to keep"
)
parser.add_argument(
    "--top_k", type=int, default=6, help="Number of documents to retrieve"
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.1,
    help="Threshold for the similarity score to consider a document as relevant",
)
parser.add_argument(
    "--num_predict_tokens", type=int, default=2000, help="Number of tokens to predict"
)
parser.add_argument(
    "--always_do_retrieval",
    type=str,
    default="true",
    help="If True, always do retrieval when answering a question.",
)
parser.add_argument(
    "--add_external_links_docs",
    type=str,
    default="false",
    help="If True, add the external links documents to the retrieval.",
)
parser.add_argument(
    "--answers_already_computed",
    type=str,
    default="false",
    help="If True, the answers that are already computed will be used to compute the scores.",
)

parser.add_argument(
    "--not_citing_source",
    type=str,
    default="false",
    help="If True, the prompt chage and the model will not cite the source of the answer.",
)



if __name__ == "__main__":
    args = parser.parse_args()
    splits_folder = args.splits_folder
    model_name = args.model_name
    test_answers_folder = args.test_answers_folder
    scores_folder = args.scores_folder

    directory = args.directory
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    embedding_model = args.embedding_model
    nb_chunks = args.nb_chunks
    top_k = args.top_k
    threshold = args.threshold
    num_predict_tokens = args.num_predict_tokens
    always_do_retrieval = str_to_bool(args.always_do_retrieval)
    add_external_links_docs = str_to_bool(args.add_external_links_docs)
    answers_already_computed = str_to_bool(args.answers_already_computed)
    not_citing_source = str_to_bool(args.not_citing_source)

    if not os.path.exists(test_answers_folder):
        os.makedirs(test_answers_folder)

    if not os.path.exists(scores_folder):
        os.makedirs(scores_folder)

    vector_base = VectorBase(
        directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        nb_chunks=nb_chunks,
    )
    

    n_splits = np.max(
        [
            int(f.split("test_fold_")[1].split(".jsonl")[0]) for f in os.listdir(splits_folder) if (f.endswith(".jsonl")
                                                                                                    and "test_fold_" in f)
        ]
    )
    print(f"Number of splits: {n_splits}")


    for i in tqdm(range(n_splits), desc="Evaluating splits"):
        questions_with_urls_path=os.path.join(
            splits_folder, f"test_fold_{i+1}.jsonl"
        )
        answers_path=os.path.join(
            test_answers_folder, f"answers_fold_{i+1}.jsonl"
        )
        scores_path=os.path.join(
            scores_folder, f"scores_fold_{i+1}.jsonl"
        )
        evaluate_rag_model(
            vector_base=vector_base,
            model_name=model_name,
            num_predict_tokens=num_predict_tokens,
            top_k=top_k,
            threshold=threshold,
            always_do_retrieval=always_do_retrieval,
            add_external_links_docs=add_external_links_docs,
            thread_id="abc123",
            questions_with_urls_path=questions_with_urls_path,
            answers_path=answers_path,
            scores_path=scores_path,
            answers_already_computed=answers_already_computed,
            not_citing_source=not_citing_source,
        )

    #load all scores and put them in a single file
    all_scores_path=os.path.join(scores_folder, "all_scores.jsonl")
    all_scores=[]
    files_scores = os.listdir(scores_folder)
    files_sorted = sorted(files_scores, key=lambda x: int(x.split("scores_fold_")[1].split(".")[0]) if "scores_fold_" in x else float('inf'))
    for scores_file in files_sorted:
        if scores_file.endswith(".jsonl") and scores_file != "all_scores.jsonl":
            score_fold=load_jsonl(os.path.join(scores_folder, scores_file))
            all_scores.extend(score_fold)

    #compute mean and std of the scores
    mean_scores = {}
    keys_metrics = all_scores[0].keys()
    for key in keys_metrics:
        scores = [score[key] for score in all_scores]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        mean_scores[key] = {
            "mean":np.round(mean_score, 2),
            "std": np.round(std_score, 2)
        }
    all_scores.append(mean_scores)
    save_jsonl(all_scores_path, all_scores)
    print(f"All scores saved in {all_scores_path}")
    


