# script to compute the URL metrics on a questions set with a RAG

import argparse
import os

from utils import evaluate_rag_model, str_to_bool

from CIT.RAGs.RAG_CIT import VectorBase

parser = argparse.ArgumentParser(
    description="Compute precision and recall for the CIT Knowledge Base."
)

parser.add_argument(
    "--directory",
    type=str,
    default="../documents/confluence_json",
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
    "--top_k", type=int, default=5, help="Number of documents to retrieve"
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.1,
    help="Threshold for the similarity score to consider a document as relevant",
)
parser.add_argument(
    "--num_predict_tokens", type=int, default=1000, help="Number of tokens to predict"
)

parser.add_argument(
    "--always_do_retrieval",
    type=str,
    default="true",
    help="If True, always do retrieval when answering a question.",
)

parser.add_argument(
    "--questions_with_urls_path",
    type=str,
    default="./src/CIT/evaluation/QA_generation/data/questions_with_urls.jsonl",
    help="Path to the JSON file containing the questions with URLs.",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="llama3.1:8b",
    help="Name of the model to answer questions and evaluate.",
)
parser.add_argument(
    "--answers_path",
    type=str,
    default="./src/CIT/evaluation/results/answers/llama3.1:8b_answers.jsonl",
    help="Path to the output answers JSON file.",
)
parser.add_argument(
    "--answers_already_computed",
    type=str,
    default="false",
    help="If True, the answers are already computed and saved in the answers_path. We just evaluate the precision and recall.",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="./src/CIT/evaluation/results/precision_recall/llama3.1:8b_precision_recall.json",
    help="Path to the output recall/precision JSON file.",
)
args = parser.parse_args()

if __name__ == "__main__":
    directory = args.directory
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    embedding_model = args.embedding_model
    nb_chunks = args.nb_chunks
    top_k = args.top_k
    threshold = args.threshold
    always_do_retrieval = args.always_do_retrieval
    always_do_retrieval = str_to_bool(always_do_retrieval)
    print(f"always_do_retrieval: {always_do_retrieval}")
    model_name = args.model_name
    num_predict_tokens = args.num_predict_tokens
    questions_with_urls_path = args.questions_with_urls_path
    answers_path = args.answers_path
    answers_already_computed = args.answers_already_computed
    answers_already_computed = str_to_bool(answers_already_computed)
    print(f"answers_already_computed: {answers_already_computed}")
    output_path = args.output_path

    if not os.path.exists(answers_path):
        os.makedirs(os.path.dirname(answers_path), exist_ok=True)
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    vector_base = VectorBase(
        directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        nb_chunks=nb_chunks,
    )

    evaluate_rag_model(
        vector_base=vector_base,
        model_name=model_name,
        num_predict_tokens=num_predict_tokens,
        top_k=top_k,
        threshold=threshold,
        always_do_retrieval=always_do_retrieval,
        thread_id="abc123",
        questions_with_urls_path=questions_with_urls_path,
        answers_path=answers_path,
        scores_path=output_path,
        answers_already_computed=answers_already_computed,
    )
    