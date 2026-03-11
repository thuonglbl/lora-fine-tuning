import os
from argparse import ArgumentParser

from judge_utils import (
    evaluate_generated_answers_parallel,
    put_real_retrieved_context_on_samples_parallel,
)
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_ollama import ChatOllama

from CIT.evaluation.utils import load_jsonl, save_jsonl
from CIT.RAGs.RAG_CIT import VectorBase

parser= ArgumentParser(
    description="Evaluate generated answers using a judging LLM."
)

parser.add_argument(
    "--directory",
    type=str,
    default="../documents/confluence_json_without_root_with_titles",
    help="Directory containing the documents from which we build the VectorBase.",
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
    "--top_k", type=int, default=6, help="Number of documents to retrieve"
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.1,
    help="Threshold for the relevance score of the retrieved documents",
)

parser.add_argument(
    "--questions_with_answers",
    type=str,
    default="./src/CIT/training/models/cv/data/test_answers/ft_02.06_with_at_least_1_url_eval_all_data/ft/answers_fold_5.jsonl",
    help="Path to the JSONL file containing questions and answers.",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="./src/CIT/evaluation/results/llm_judge_examples/ft_02.06_with_at_least_1_url_eval_all_data/ft/answers_fold_5_binary.jsonl",
    help="Path to the JSONL file the questions and answers woth their ratings.",
)

parser.add_argument(
    "--judge_model",
    type=str,
    default="llama3.1:8b",
    help="The model to use for judging the answers. Default is 'llama3.1:8b'.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    directory = args.directory
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    embedding_model = args.embedding_model
    top_k = args.top_k
    threshold = args.threshold
    questions_with_answers = load_jsonl(args.questions_with_answers)
    
    llm_judge = ChatOllama(
        model=args.judge_model,
        temperature=0.0
    )

    vector_base= VectorBase(
        directory=directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model
    )


    # Load the vector base retriever

    #load retriever to have the real retrieved context
    top_k_pre_rerank = 15
    compression_retriever=retriever = vector_base.vectorstore.as_retriever(
        search_kwargs={"k": top_k_pre_rerank}
    )
    compressor = FlashrankRerank(top_n=top_k, model="ms-marco-MiniLM-L-12-v2")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    mapping_id_paths = vector_base.mapping_id_paths
    # Get the retrieved context for each question
    questions_with_answers = put_real_retrieved_context_on_samples_parallel(
        questions_with_answers,
        compression_retriever,
        threshold=threshold,
        mapping_id_paths=mapping_id_paths
    )


    # Evaluate the generated answers
    results = evaluate_generated_answers_parallel(
        questions_with_answers, llm_judge, num_workers=4
    )

    # Save the results to the output path
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    save_jsonl(args.output_path, results)
