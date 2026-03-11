# RAG python script for Wallis laws

import glob
import os
import pickle
from argparse import ArgumentParser
from collections import Counter

import numpy as np
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument(
    "--directory",
    type=str,
    default="../transformed_documents",
    help="Directory containing the documents to be indexed",
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
    "--top_k", type=int, default=6, help="Number of documents to retrieve"
)
parser.add_argument(
    "--model_name",
    type=str,
    default="llama3.1:8b",
    help="Model name to use for the RAG model",
)
parser.add_argument(
    "--num_predict_tokens", type=int, default=300, help="Number of tokens to predict"
)
parser.add_argument(
    "--reranking", type=bool, default=True, help="Whether to use reranking or not"
)
parser.add_argument(
    "--always_do_retrieval",
    type=bool,
    default=True,
    help="Whether to always do retrieval or not",
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
    default="VectorBase.pkl",
    help="Path to the vectorbase to load",
)


# transform str list to list
def transform_list(x):
    if isinstance(x, str):
        return list(map(int, x.strip("[]").split(",")))
    return x


class VectorBase:
    def __init__(
        self,
        directory,
        chunk_size=1000,
        chunk_overlap=50,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        nb_chunks=-1,
    ):
        self.directory = directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.chunks = self.load_and_chunk_documents()
        print(f"Loaded {len(self.chunks)} chunks")
        self.chunks = self.add_title_to_chunks()
        self.chunks = self.chunks[:nb_chunks]
        self.vectorstore = self.create_vectorstore()

    def load_and_chunk_documents(self):
        file_paths = glob.glob(os.path.join(self.directory, "*.txt"))
        documents = []

        for file_path in tqdm(file_paths, total=len(file_paths)):
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)

        # Split documents into chunks

        if isinstance(self.chunk_size, list):  # multiple chunk sizes
            print("multiple chunk sizes")
            print(self.chunk_size)
            chunks = []
            for ch_size in self.chunk_size:
                ch_size = int(ch_size)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=ch_size, chunk_overlap=self.chunk_overlap
                )
                chunks_spe = text_splitter.split_documents(documents)
                chunks.extend(chunks_spe)
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
        return chunks

    def add_title_to_chunks(
        self,
    ):  # add the title of the document as the first line of the chunk. It makes retrieval easier
        file_paths = glob.glob(os.path.join(self.directory, "*.txt"))
        mapping = {}
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                mapping[file_path] = text.split("\n")[
                    0
                ]  # because preprocessing such that first line is title
        for chunk in self.chunks:
            chunk.metadata["title"] = mapping[chunk.metadata["source"]]

            if (
                not chunk.metadata["title"][:4] == chunk.page_content[:4]
            ):  # to avoid repeating title in chunks that already have the title
                chunk.page_content = chunk.metadata["title"] + "\n" + chunk.page_content

        return self.chunks

    def create_vectorstore(self):
        print("Creating vectorstore...")
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        vectorstore = FAISS.from_documents(self.chunks, embeddings)
        return vectorstore


def build_graph(
    vector_base: VectorBase,
    model_name="llama3.1:8b",
    num_predict=100,
    top_k=3,
    reranking=True,
    always_do_retrieval=False,
):
    """
    Build a RAG with a langchain graph, from a vectorbase
    """

    llm = ChatOllama(
        model=model_name, num_predict=num_predict, num_ctx=6096, temperature=0
    )  # set the tempearture to 0 for reproducability

    if reranking:  # which should be done
        retriever = vector_base.vectorstore.as_retriever(
            search_kwargs={"k": 15}
        )  # we take the 15 most relevant documents before reranking
        compressor = FlashrankRerank(top_n=top_k, model="ms-marco-MiniLM-L-12-v2")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

    @tool(response_format="content_and_artifact")  # langchain tool
    def retrieve(query: str):
        """Retrieve documents related to a user query to help answer a question.
        Args:
            query (str): user question or query.
        Returns:
            Tuple[str, List[Document]]: retrieved documents."""
        if reranking:  # it should be done
            retrieved_docs = compression_retriever.invoke(query)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            cleaned_docs = []
            for doc in retrieved_docs:
                metadata = doc.metadata
                relevance_score = metadata.pop("relevance_score")
                doc.metadata = metadata
                cleaned_docs.append((doc, relevance_score.astype(np.float64)))
            retrieved_docs = cleaned_docs
        else:  # may be deprecated
            retrieved_docs = (
                vector_base.vectorstore.similarity_search_with_relevance_scores(
                    query, k=top_k
                )
            )
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nScore: {score}\nContent: {doc.page_content}")
                for doc, score in retrieved_docs
            )
        return serialized, retrieved_docs

    def query_or_respond(state: MessagesState):
        """Test if the user query needs retrieval or if the model can answer directly
        e.g it is useless to make retrieval whan the user says 'Hello')
        """
        if not always_do_retrieval:  # ask the llm id retrieval is required
            user_input = state["messages"][-1].content
            """Generate tool call for retrieval or respond."""
            prompt_need_retrieval = (
                f"Does this user input need a retrieval step of RAG ?\
                For example if it is just a greeting, there is no need for further information, you should answer False.\
                If it is a question or a statement regarding the canton du Valais, administrative task or procedure, some public intitutions, or texts of laws, you should answer True.\
                    Answer only by True or False\nUser input: ?{user_input}"
            )
            response = llm.invoke(prompt_need_retrieval)
            # print(f"Need retrieval: {response.content}")
            if response.content == "True":
                llm_with_tools = llm.bind_tools([retrieve])
                # print("retrieval")
            else:
                print("no retrieval")
                llm_with_tools = llm.bind_tools([])
        else:  # always do retrieval
            llm_with_tools = llm.bind_tools(
                [retrieve], tool_choice=retrieve
            )  # currently tool_choice is not supported for models others than OpenAI, so it does not alwaws work

        response = llm_with_tools.invoke(
            state["messages"]
        )  # retrieval tool might be called in the state

        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}

    # Generate a response using the retrieved content.
    def generate(state: MessagesState):
        """Generate an answer from retrieved documents."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[
            ::-1
        ]  # get documents retrieved from the last tool call (retrieval call)
        # Format into prompt
        sources = [
            doc[0].metadata["source"]
            for tool_message in tool_messages
            for doc in tool_message.artifact  # Access the document objects inside artifact
        ]
        sources = Counter(sources)

        primary_source = Counter(sources).most_common(1)[0][
            0
        ]  # most represented source in the top_k documents
        with open(primary_source, "r") as f:
            primary_source_content = (
                f.read()
            )  # we will add the entire document in the context prompt
        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        docs_content = (
            f"Primary document: {primary_source_content}" + "\n\n" + docs_content
        )  # add most relevant document to the context
        system_message_content = (
            "You are an assistant working for the canton du Valais in Switzerland.\
        You work as a RAG. I'll give you text of laws of the canton, written in french.\
        Answer in french only within the context of the canton du Valais and with the context I give you.\
        Always cite from which law and article your answer comes from. The answer should remain concise."
            "\n\nContext:\n"
            f"{docs_content}"
        )

        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [
            SystemMessage(system_message_content)
        ] + conversation_messages  # add new prompt to conversation history

        # call the LLM
        response = llm.invoke(prompt)
        return {"messages": [response]}

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    tools = ToolNode([retrieve])
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    memory = MemorySaver()  # allows to keep old messages in memory. Not ideal as it retains a lot of message. More work on this has been done in the CIT case
    graph = graph_builder.compile(checkpointer=memory)
    return graph


def chat_with_rag(graph, config):
    """Take user input as entry to the graph and launch conversation with the RAG"""
    print("RAG Chatbot is ready! Type 'exit' to quit.")

    while True:
        input_message = input("\nYou: ")
        if input_message.lower() == "exit":
            break
        input_message = input_message.replace("'", " ").replace("’", " ")
        input_message = input_message
        for step in graph.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="values",
            config=config,
        ):
            step["messages"][-1].pretty_print()


def get_answers_from_rag(graph, config, questions):
    """Answers to a list of questions with a RAG (langchain graph + config)"""
    answers = []
    for question in questions:
        input_message = question
        input_message = input_message.replace("'", " ").replace("’", " ")
        for step in graph.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="values",
            config=config,
        ):  # go through the graph
            answers.append(step["messages"][-1].content)
    return answers


class RAGv3:
    def __init__(
        self,
        vector_base: VectorBase,
        model_name="llama3.1:8b",
        num_predict=100,
        top_k=3,
        reranking=True,
        always_do_retrieval=False,
        thread_id="abc123",
    ):
        self.vector_base = vector_base
        self.graph = build_graph(
            vector_base=vector_base,
            model_name=model_name,
            num_predict=num_predict,
            top_k=top_k,
            reranking=reranking,
            always_do_retrieval=always_do_retrieval,
        )

        self.config = {"configurable": {"thread_id": thread_id}}

    def chat(self):
        """
        Launch the chat with the RAG
        """
        chat_with_rag(self.graph, self.config)


if __name__ == "__main__":
    args = parser.parse_args()
    DIRECTORY = args.directory
    CHUNK_SIZE = args.chunk_size
    CHUNK_SIZE = transform_list(CHUNK_SIZE)

    CHUNK_OVERLAP = args.chunk_overlap
    EMBEDDING_MODEL = args.embedding_model
    NB_CHUNKS = args.nb_chunks
    RERANKING = args.reranking
    ALWAYS_DO_RETRIEVAL = args.always_do_retrieval

    TOP_K = args.top_k
    MODEL_NAME = args.model_name
    NUM_PREDICT_TOKENS = args.num_predict_tokens

    if args.load_default_vectorbase:
        print("Loading default vectorbase at path: ", args.vectorbase_path)
        vector_base = pickle.load(open(args.vectorbase_path, "rb"))
    else:
        vector_base = VectorBase(
            directory=DIRECTORY,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            embedding_model=EMBEDDING_MODEL,
            nb_chunks=NB_CHUNKS,
        )

    RAG = RAGv3(
        vector_base=vector_base,
        model_name=MODEL_NAME,
        num_predict=NUM_PREDICT_TOKENS,
        top_k=TOP_K,
        reranking=RERANKING,
        always_do_retrieval=ALWAYS_DO_RETRIEVAL,
    )

    RAG.chat()
