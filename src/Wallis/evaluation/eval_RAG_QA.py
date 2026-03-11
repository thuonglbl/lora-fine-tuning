# script to evaluate a RAG pipeline, with a LLM as judge method, on questions that contain reference answers

import json
import logging
import os
import pickle
from argparse import ArgumentParser

import numpy as np
from langchain_ollama import ChatOllama
from QA_utils import evaluate_generated_answers_parallel, get_answers_from_rag

from Wallis.RAGs.RAGv3 import RAGv3, VectorBase

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

parser = ArgumentParser()
parser.add_argument(
    "--directory",
    type=str,
    default="../transformed_documents",
    help="Directory containing the documents to be indexed",
)
parser.add_argument(
    "--load_default_vectorbase",
    type=bool,
    default=True,
    help="Whether to load the default vectorbase or not",
)
parser.add_argument(
    "--vectorbase_path",
    type=str,
    default="../RAGs/VectorBase.pkl",
    help="Path to the vectorbase to load",
)

parser.add_argument(
    "--chunk_size",
    default=[500, 1000],
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
    "--top_k", type=int, default=5, help="Number of documents to retrieve"
)
parser.add_argument(
    "--model_name",
    type=str,
    default="llama3.3",
    help="Model name to use for the RAG model",
)
parser.add_argument(
    "--num_predict_tokens", type=int, default=300, help="Number of tokens to predict"
)
parser.add_argument(
    "--reranking", type=bool, default=True, help="Whether to use reranking or not"
)
parser.add_argument(
    "--questions_path",
    type=str,
    default="../data/intermediate_results/test/retrieved_qu_1703_test.jsonl",
    help="Path to the generated questions test set, \
the questions should be contain dictionnaries with keys 'question', 'answer':reference answer, and 'title': title of the document from which the question comes",
)

parser.add_argument(
    "--recompute_errors",
    type=bool,
    default=False,
    help="Whether to recompute the errors in the answers",
)

parser.add_argument(
    "--judge_model_name",
    type=str,
    default="mistral-small",
    help="Model name to use to compare answers",
)
parser.add_argument(
    "--few_shots",
    type=bool,
    default=False,
    help="Whether to use few shots or not in evlauation",
)

parser.add_argument(
    "--answers_path",
    type=str,
    default="../data/results/retrieved_qu/base_llama3.3.jsonl",
    help="Path to save the answers of the RAG model",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="../data/results/retrieved_qu/base_llama3.3_judge-mistral_few_shots.jsonl",
    help="Path to save the answers of the RAG model",
)



if __name__ == "__main__":
    args = parser.parse_args()
    directory = args.directory
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    embedding_model = args.embedding_model
    nb_chunks = args.nb_chunks
    top_k = args.top_k
    model_name = args.model_name
    num_predict_tokens = args.num_predict_tokens
    reranking = args.reranking
    questions_path = args.questions_path
    recompute_errors = args.recompute_errors
    judge_model_name = args.judge_model_name
    few_shots = args.few_shots
    answers_path = args.answers_path
    output_path = args.output_path

    if args.load_default_vectorbase:  # load the default vectorabase, stored as a pickle file, allows to speed up the process
        print("Loading default vectorbase at path: ", args.vectorbase_path)
        vector_base = pickle.load(open(args.vectorbase_path, "rb"))
    else:
        vector_base = VectorBase(
            directory,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            nb_chunks=nb_chunks,
        )
    RAG = RAGv3(
        vector_base,
        model_name=model_name,
        num_predict=num_predict_tokens,
        top_k=top_k,
        reranking=reranking,
        always_do_retrieval=True,
        thread_id="abc123",
    )
    print("RAG model built")
    graph = RAG.graph
    config = RAG.config

    # open questions
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    test = False
    if test:
        questions = questions[-2:]

    print(f"Loaded {len(questions)} questions")
    if os.path.exists(
        answers_path
    ):  # check if there are existing answers at the output location
        with open(answers_path, "r", encoding="utf-8") as f:
            answers = json.load(f)
        print(f"Loaded {len(answers)} answers already computed")

    else:
        answers = []
    answers = get_answers_from_rag(
        graph,
        config,
        questions,
        answers_path,
        answers,
        recompute_errors=recompute_errors,
    )
    print(f"Computed {len(answers)} answers")
    # store answers as checkpoints
    # check proportions of errors in the answers
    err = 0
    for answer in answers:
        if answer["RAG_answer"] == "Error":
            err += 1
    print(f"Proportion of errors in the answers: {err / len(answers)}")

    # save answers
    if not os.path.isdir(os.path.dirname(answers_path)):
        print(f"Creating directory {os.path.dirname(answers_path)}")
        os.makedirs(os.path.dirname(answers_path))
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False)
    print(f"Answers saved to {answers_path}")

    # evaluate the answers

    llm_judge = ChatOllama(
        model=judge_model_name, temperature=0
    )  # the llm that will grade the generated answers
    results = evaluate_generated_answers_parallel(
        answers, llm_judge, few_shots=few_shots
    )
    # save results with ratings
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"Ratings saved to {output_path}")
    mean_rating = np.nanmean([result["rating"] for result in results])
    std_rating = np.nanstd([result["rating"] for result in results])
    evaluation_errors = np.sum(np.isnan([result["rating"] for result in results]))
    print(
        f"proportion of evaluation errors: {evaluation_errors / len(results)} (includes questions that were not answered)"
    )
    print(f"Mean rating: {mean_rating}")
    print(f"Std rating: {std_rating}")
