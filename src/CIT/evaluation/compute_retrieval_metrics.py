# script to compute the retrieval component metrics on a list of questions

import argparse
import json

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from utils import compute_retrieval_stats, load_jsonl

from CIT.RAGs.utils import VectorBase

parser = argparse.ArgumentParser(
    description="Compute precision and recall for the retrieval of CIT Knowledge Base."
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
    "--questions_with_urls_path",
    type=str,
    default="./src/CIT/evaluation/QA_generation/data/questions_with_urls_benj.jsonl",
    help="Path to the JSON file containing the questions with URLs.",
)

parser.add_argument(
    "--output_path",
    type=str,
    default="./src/CIT/evaluation/results/retrieval/base.json",
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
    questions_with_urls_path = args.questions_with_urls_path
    output_path = args.output_path

    # load vector base from documents and instantiate retriever
    vector_base = VectorBase(
        directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        nb_chunks=nb_chunks,
    )

    top_k_pre_rerank = 15 if top_k < 15 else top_k + 10
    retriever = vector_base.vectorstore.as_retriever(
        search_kwargs={"k": top_k_pre_rerank}
    )
    compressor = FlashrankRerank(top_n=top_k, model="ms-marco-MiniLM-L-12-v2")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    # load questions
    questions_with_urls = load_jsonl(questions_with_urls_path)
    print(f"Length of questions_with_urls: {len(questions_with_urls)}")

    # compute metrics
    mean_precision, mean_recall, average_precision, average_recall = (
        compute_retrieval_stats(questions_with_urls, compression_retriever)
    )
    print(f"Mean precision: {mean_precision:2f}")
    print(f"Mean recall: {mean_recall:2f}")
    print(f"Average precision: {average_precision:2f}")
    print(f"Average recall: {average_recall:2f}")

    with open(output_path, "w") as f:
        json.dump(
            {
                "mean_precision": mean_precision,
                "mean_recall": mean_recall,
                "average_precision": average_precision,
                "average_recall": average_recall,
            },
            f,
        )
