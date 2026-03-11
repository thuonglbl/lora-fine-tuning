# script to create a RAG for the JIRA documents. For now, the JIRA documents created are not very useful. It was use as an experiment.
#The JIRA tickets are not very useful
# using langchain and langgraph


from argparse import ArgumentParser

from utils_jira import RAG_JIRA, VectorBase_JIRA

parser = ArgumentParser()
parser.add_argument(
    "--directory",
    type=str,
    default="./src/CIT/scraping/JIRA/jira_documents",
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
    help="Threshold for the relevance score of the retrieved documents",
)


parser.add_argument(
    "--model_name",
    type=str,
    default="user_ft0505_sd_urls",
    help="Model name to use for the RAG model",
)
parser.add_argument(
    "--num_predict_tokens",
    type=int,
    default=1000,
    help="Maximum number of tokens generated in the answer",
)
parser.add_argument(
    "--keep_in_mind_last_n_messages",
    type=int,
    default=4,
    help="Number of messages to keep in mind for the RAG model",
)
parser.add_argument(
    "--always_do_retrieval", default=False, help="Whether to always do retrieval or not"
)
parser.add_argument(
    "--verbose", default=True, help="Whether to print the retrieved documents or not"
)


# transform str list to list
def transform_list(x):
    if isinstance(x, str):
        return list(map(int, x.strip("[]").split(",")))
    return x


def str_to_bool(x):
    if isinstance(x, str):
        if x.lower() == "true":
            return True
        elif x.lower() == "false":
            return False
    return x


if __name__ == "__main__":
    args = parser.parse_args()
    DIRECTORY = args.directory
    CHUNK_SIZE = args.chunk_size
    CHUNK_SIZE = transform_list(CHUNK_SIZE)

    CHUNK_OVERLAP = args.chunk_overlap
    EMBEDDING_MODEL = args.embedding_model
    NB_CHUNKS = args.nb_chunks
    ALWAYS_DO_RETRIEVAL = args.always_do_retrieval
    ALWAYS_DO_RETRIEVAL = str_to_bool(ALWAYS_DO_RETRIEVAL)
    VERBOSE = args.verbose
    VERBOSE = str_to_bool(VERBOSE)

    TOP_K = args.top_k
    THRESHOLD = args.threshold
    MODEL_NAME = args.model_name
    NUM_PREDICT_TOKENS = args.num_predict_tokens
    KEEP_IN_MIND_LAST_N_MESSAGES = args.keep_in_mind_last_n_messages

    # Create the vectorstore
    vector_base = VectorBase_JIRA(
        directory=DIRECTORY,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedding_model=EMBEDDING_MODEL,
        nb_chunks=NB_CHUNKS,
    )
    # Create the RAG model
    # RAGv3 is a class that creates a RAG model using langgraph
    RAG = RAG_JIRA(
        vector_base=vector_base,
        model_name=MODEL_NAME,
        num_predict=NUM_PREDICT_TOKENS,
        user_prompt="Answer my questions",
        top_k=TOP_K,
        threshold=THRESHOLD,
        keep_in_mind_last_n_messages=KEEP_IN_MIND_LAST_N_MESSAGES,
        always_do_retrieval=ALWAYS_DO_RETRIEVAL,
        verbose=VERBOSE,
    )

    RAG.chat()
