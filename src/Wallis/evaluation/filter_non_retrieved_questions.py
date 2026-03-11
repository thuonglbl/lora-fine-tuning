# script to filter non retrieved questions (with top_k=5) from a question list

import json
import os
import pickle
from argparse import ArgumentParser

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from QA_utils import filter_non_retrieved_questions

parser = ArgumentParser()
parser.add_argument(
    "--questions_path",
    type=str,
    default="../data/intermediate_results/retrieved_qu_1703.jsonl",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="../data/intermediate_results/retrieved_qu_1703.jsonl",
)
args = parser.parse_args()


if __name__ == "__main__":
    questions_path = args.questions_path
    output_path = args.output_path
    filtered_questions = json.load(open(questions_path, "r"))

    if not os.path.isdir(os.path.dirname(output_path)):
        print(f"Creating directory {os.path.dirname(output_path)}")
        os.makedirs(os.path.dirname(output_path))

    # build vector base and retrieval tools
    vectorbase_retrieving = pickle.load(open("../RAGs/VectorBase.pkl", "rb"))
    retriever = vectorbase_retrieving.vectorstore.as_retriever(search_kwargs={"k": 15})
    compressor = FlashrankRerank(top_n=5, model="ms-marco-MiniLM-L-12-v2")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    filtered_questions = filter_non_retrieved_questions(
        filtered_questions, vectorbase_retrieving, compression_retriever, reranking=True
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_questions, f, ensure_ascii=False)
    print(f"Generated questions saved to {output_path}")
