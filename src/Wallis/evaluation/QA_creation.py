# script to create a dataset of questions and answers automatically using an LMM

import json
import logging

# move to another folder
import pickle
from argparse import ArgumentParser

from Wallis.RAGs.RAGv3 import VectorBase

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)  # to avoid printing info of llm calls in the terminal

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_ollama import ChatOllama
from QA_utils import (
    filter_non_retrieved_questions,
    filter_questions,
    generate_questions_for_chunks_parallel,
    grade_generated_questions,
)

parser = ArgumentParser()
parser.add_argument("--chunk_size", type=int, default=6000)
parser.add_argument("--chunk_overlap", type=int, default=1)

parser.add_argument("--min_groundness", type=int, default=3)
parser.add_argument("--min_relevance", type=int, default=3)

parser.add_argument("--model_name_creation", type=str, default="llama3.1:8b")
parser.add_argument("--model_name_grading", type=str, default="llama3.1:8b")

parser.add_argument(
    "--output_path_all_questions",
    type=str,
    default="intermediate_results/all_qu_1703.jsonl",
)
parser.add_argument(
    "--output_path_retrieved_questions",
    type=str,
    default="intermediate_results/retrieved_qu_1703.jsonl",
)

args = parser.parse_args()

if __name__ == "__main__":
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    min_groundness = args.min_groundness
    min_relevance = args.min_relevance
    model_name_grading = args.model_name_grading
    model_name_creation = args.model_name_creation
    output_path_all_questions = args.output_path_all_questions
    output_path_retrieved_questions = args.output_path_retrieved_questions

    vectorebase = VectorBase(
        directory="../transformed_documents",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = vectorebase.chunks
    llm = ChatOllama(model=model_name_creation, num_predict=700, temperature=0)
    generated_questions = generate_questions_for_chunks_parallel(
        chunks, llm
    )  # generate the QA from a list of chunks

    llm_groundness = ChatOllama(
        model=model_name_grading, num_predict=200, temperature=0
    )
    llm_relevance = ChatOllama(model=model_name_grading, num_predict=200, temperature=0)

    # grade and filter the generated questions
    print("Grading generated questions...")
    graded_questions = grade_generated_questions(
        generated_questions, llm_groundness, llm_relevance
    )
    filtered_questions = filter_questions(
        graded_questions, min_groundness=min_groundness, min_relevance=min_relevance
    )

    # save questions
    with open(output_path_all_questions, "w", encoding="utf-8") as f:
        json.dump(graded_questions, f, ensure_ascii=False)
    print(f"All questions saved to {output_path_all_questions}")

    # save only retrieved questions (with top_k=5) to a different path
    vectorbase_retrieving = pickle.load(open("../RAGs/VectorBase.pkl", "rb"))
    retriever = vectorbase_retrieving.vectorstore.as_retriever(search_kwargs={"k": 15})
    compressor = FlashrankRerank(top_n=5, model="ms-marco-MiniLM-L-12-v2")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    filtered_questions = filter_non_retrieved_questions(
        filtered_questions, vectorbase_retrieving, compression_retriever, reranking=True
    )
    with open(output_path_retrieved_questions, "w", encoding="utf-8") as f:
        json.dump(filtered_questions, f, ensure_ascii=False)
    print(
        f"Generated and filtered questions saved to {output_path_retrieved_questions}"
    )
