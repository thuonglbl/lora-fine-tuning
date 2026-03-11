# script to create a RAG for the Confluence documents
# using langchain and langgraph

#This version of the RAG is only used for evaluation, please use the RAG from utils.py for real inference.

# Steps:
# 1. Load the documents from the directory
# 2. Chunk the documents into smaller pieces
# 3. Create a vectorstore from the chunks
# 4. Create a retriever from the vectorstore
# 5. Create a RAG graph using langgraph
# 6. Create a chatbot using the RAG graph

import glob
import json
import os
import re
from argparse import ArgumentParser
from collections import Counter
from uuid import uuid4

from CIT.config import settings

import numpy as np
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.tools import tool
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


def extract_confluence_urls(text):
    """Finds all URLs starting with the configured Confluence base URL."""
    text = text.replace("](", "] (")
    base_url_regex = settings.CONFLUENCE_BASE_URL.replace(".", r"\.")
    url_pattern = re.compile(rf"{base_url_regex}/[^\s,\)]+")
    all_matches = url_pattern.findall(text)
    cleaned_matches = [
        match.replace("&src=contextnavpagetreemode", "")
        .replace("?src=contextnavpagetreemode", "")
        .replace("pages/viewpage.action?pageId=", "spaces/CORPORATEITKNOWLEDGEBASE/pages/")
        .strip(",.)];").split("#")[0]
        for match in all_matches
    ]

    return cleaned_matches


def url_harmonization(url, url_to_standard_url_mapping):
    """
    Harmonize a URL to a standard URL.
    Args:
        url: URL to harmonize (str)
        url_to_standard_url_mapping: dictionary with keys as URLs and values as standard URLs
         with template "https://confluence.yourcompany.com/spaces/CORPORATEITKNOWLEDGEBASE/pages/{id}"
    Returns:
        standard_url: standard URL
    """
    if url in url_to_standard_url_mapping:
        return url_to_standard_url_mapping[url]
    else:
        print(f"URL not in mapping: {url}")
        return url


parser = ArgumentParser()
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
    help="Threshold for the relevance score of the retrieved documents",
)

parser.add_argument(
    "--add_external_links_docs",
    type=str,
    default="false",
    help="Whether to add the documents that are quoted in the primary retrieved documents",
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
    default=6,
    help="Number of messages to keep in mind for the RAG model",
)
parser.add_argument(
    "--always_do_retrieval", default=True, help="Whether to always do retrieval or not"
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


class VectorBase:
    """
    Class to create a vectorstore from a directory of documents.
    Args:
        directory (str): Directory containing the documents to be indexed.
        chunk_size (int): Size of the chunks to split the documents into.
        chunk_overlap (int): Overlap between chunks.
        embedding_model (str): Huggingface model to use for embeddings.
        nb_chunks (int): Number of chunks to keep  (param used to debug, default: -1 means all chunks)
    """

    def __init__(
        self,
        directory,
        chunk_size=1000,
        chunk_overlap=50,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        nb_chunks=-1,
    ):
        self.directory = directory
        print(f"absolute path: {os.path.abspath(directory)}")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.chunks = self.load_and_chunk_documents()
        print(f"Loaded {len(self.chunks)} chunks")
        self.chunks = self.chunks[:nb_chunks]
        self.vectorstore = self.create_vectorstore()
        with open(directory + "/mappings/mapping_id_paths.json", "r") as f:
            mapping_id_paths = json.load(f)
            self.mapping_id_paths = mapping_id_paths
        with open(directory + "/mappings/mapping_original_urls_fixed_urls.json", "r") as f:
            mapping_original_urls_fixed_urls = json.load(f)
            self.mapping_original_urls_fixed_urls = mapping_original_urls_fixed_urls

    def load_and_chunk_documents(self):
        file_paths = glob.glob(os.path.join(self.directory, "*.json"))
        documents = []
        for file_path in file_paths:
            doc_dict = json.load(open(file_path, "r"))
            metadata = {key: doc_dict[key] for key in doc_dict if key != "content"}
            metadata.update({"source": file_path})
            doc = Document(
                page_content=doc_dict["content"],
                metadata=metadata,
            )
            documents.append(doc)
        print(f"Number of documents: {len(documents)}")


        # Split documents into chunks
        if isinstance(self.chunk_size, list):
            print(f"chunk_size: {self.chunk_size}")
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

    def create_vectorstore(self):
        print("Creating vectorstore...")
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        vectorstore = FAISS.from_documents(self.chunks, embeddings)
        return vectorstore


def build_graph(
    vector_base: VectorBase,
    model_name="llama3.1:8b",
    num_predict=100,
    top_k=6,
    threshold=0.1,
    keep_in_mind_last_n_messages=2,
    always_do_retrieval=False,
    verbose=False,
    add_external_links_docs=False,
    not_citing_source=False,
):
    """Build the RAG graph.
    Args:
        vector_base (VectorBase): VectorBase object containing the vectorstore.
        model_name (str): Model name to use for the RAG model.
        num_predict (int): Number of tokens to predict.
        top_k (int): Number of documents to retrieve.
        threshold (float): Threshold for the relevance score of the retrieved documents.
        always_do_retrieval (bool): Whether to always do retrieval or not.
        verbose (bool): Whether to print the retrieved documents or not.
    Returns:
        graph (StateGraph): RAG graph."""
    print(f"Threshold: {threshold}")
    print(f"Add external links docs: {add_external_links_docs}")

    llm = ChatOllama(
        model=model_name,
        num_predict=num_predict,
        num_ctx=10000,
        temperature=0,
        seed=69,
    )

    # reranking from the top 15 retrieved documents
    top_k_pre_rerank = 15 if top_k < 15 else top_k + 5
    # Step 1: Create a retriever from the vectorstore
    retriever = vector_base.vectorstore.as_retriever(
        search_kwargs={"k": top_k_pre_rerank}
    )
    # Step 2: Create a compression retriever
    compressor = FlashrankRerank(top_n=top_k, model="ms-marco-MiniLM-L-12-v2")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    # load mapping_id_paths (useful to get the original documents if needed)
    mapping_id_paths = vector_base.mapping_id_paths

    url_to_standard_url_mapping = vector_base.mapping_original_urls_fixed_urls

    # define the trimmer to manage the memory
    trimmer = trim_messages(strategy="last", max_tokens=keep_in_mind_last_n_messages, token_counter=len)

   
    # Step 3: Define the tools
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve documents related to a user query to help answer a question.
        Args:
            query (str): user question or query.
        Returns:
            Tuple[str, List[Document]]: retrieved documents."""
        retrieved_docs = compression_retriever.invoke(query)
        if verbose:
            print(query)
            for doc in retrieved_docs:
                print(doc.metadata["title"])
                print(doc.metadata["relevance_score"])
                print("--------------------------")

        parents_id = Counter([doc.metadata["parent"] for doc in retrieved_docs])
        index_most_relevant_doc = np.argmax(
            [doc.metadata["relevance_score"] for doc in retrieved_docs]
        )
        primary_source = retrieved_docs[index_most_relevant_doc].metadata["source"]
        sources_id = Counter([doc.metadata["id"] for doc in retrieved_docs])

        # if principal parent is present at least 3 times, we load it
        add_parent = False
        if parents_id.most_common(1)[0][1] > 2:
            primary_parent_id = parents_id.most_common(1)[0][0]
            if (primary_parent_id in mapping_id_paths) and (
                primary_parent_id not in sources_id
            ):
                add_parent = True
                with open(mapping_id_paths[primary_parent_id], "r") as f:
                    primary_parent_content = json.load(f)

        scale = np.max([doc.metadata["relevance_score"] for doc in retrieved_docs])
        real_threshold = threshold * scale
        # filter out documents with relevance score < 0.3*scale
        retrieved_docs = [
            doc
            for doc in retrieved_docs
            if doc.metadata["relevance_score"] >= real_threshold
        ]
        if verbose:
            print(f"threshold: {real_threshold}")


        if add_external_links_docs:
            # add confluence pages cited in the retrieved context
            outgoing_ids = [doc.metadata["outgoing_page_ids"] for doc in retrieved_docs]
            # remove duplicates
            outgoing_ids = list(set([item for sublist in outgoing_ids for item in sublist]))
            # remove the ids of the documents already in the retrieved context
            outgoing_ids = [id for id in outgoing_ids if id not in sources_id]
            # load the documents
            for id in outgoing_ids:
                if id in mapping_id_paths:
                    with open(mapping_id_paths[id], "r") as f:
                        doc = json.load(f)
                        retrieved_docs.append(
                            Document(
                                page_content=doc["content"],
                                metadata={
                                    "id": id,
                                    "title": doc["title"],
                                    "url": doc["url"],
                                    "source": mapping_id_paths[id],
                                    "original_title": doc["original_title"],
                                },
                            )
                        )
                        print(
                            f"Added document {doc['title']} with id {id} to the retrieved context"
                        )

        # add full content of the primary source
        with open(primary_source, "r") as f:
            primary_source = json.load(f)

        template_url = settings.CONFLUENCE_PAGE_TEMPLATE

        # load the whole content of the primary source
        primary_source_content = primary_source["content"]
        primary_source_title = primary_source["title"]
        primary_source_url = template_url.format(id=primary_source["id"])
        # Format retrieved context made of full primary document and chunks of secondary documents
        secondary_docs = [
            doc for doc in retrieved_docs if doc.metadata["id"] != primary_source["id"]
        ]

        if model_name == "random_urls_model_baseline":
            #only give the urls of the retrieved documents
            retrieved_context = (
                f"Document URL: {primary_source_url}\n"
                + "\nOther documents:\n"
                + "\n\n".join(
                    f"Document {i + 2} URL: {template_url.format(id=doc.metadata['id'])}"
                    for i, doc in enumerate(secondary_docs)
                )
            )
            add_parent = False
        else:  # give the full content of the primary document and chunks of secondary documents

            retrieved_context = (
                f"Primary document title: {primary_source_title}\n"
                f"Document URL: {primary_source_url}\n"
                f"Document content: {primary_source_content}"
                + "\nOther documents:\n"
                + "\n".join(
                    f"Title: {doc.metadata['original_title']}\n"
                    f"URL: {template_url.format(id=doc.metadata['id'])}\n"
                    f"Document content: {doc.page_content}"
                    for doc in retrieved_docs
                    if doc.metadata["source"] != primary_source
                )
            )
        if add_parent:  # add the parent document
            retrieved_context += (
                "\nParent document:\n"
                f"Title: {primary_parent_content['title']}\n"
                f"URL: {primary_parent_content['url']}\n"
                f"{primary_parent_content['content']}"
            )
        cleaned_docs = []
        for doc in retrieved_docs:
            metadata = doc.metadata
            if "relevance_score" in metadata:
                relevance_score = metadata.pop("relevance_score")
                relevance_score = relevance_score.astype(np.float64)
            else:
                relevance_score = np.float64(np.nan)
            doc.metadata = metadata
            cleaned_docs.append((doc, relevance_score))

        return retrieved_context, cleaned_docs

    def query_or_respond(state: MessagesState):
        """Check if the user input needs a retrieval step or not.
        Args:
            state (MessagesState): State of the conversation.
        Returns:
            MessagesState: State of the conversation with the retrieved documents.
        """

        # Get the last user input
        user_input = state["messages"][-1].content
        if not always_do_retrieval:
            """Generate tool call for retrieval or respond."""
            prompt_needs_retrieval = f"Does this user input need a retrieval step of RAG ?\
                For example if it is just a greeting, there is no need for further information, you should answer False.\
                If it is a question or a statement regarding an issue or something to solve you should answer True. Only answer True or False{user_input}"
            response = llm.invoke(prompt_needs_retrieval)

            if response.content == "True":
                llm_with_tools = llm.bind_tools([retrieve])
            else:
                print("no retrieval")
                llm_with_tools = llm.bind_tools([])
            #trim messages to the last N messages

            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            trimmed_messages = trimmer.invoke(
                conversation_messages
            )
            if verbose:
                print("trimmed messages")
                for i,message in enumerate(trimmed_messages):
                    #check if the message is a tool call
                    print(f"Message {i}: {message}")
                    print("--------------------------")
            
            response = llm_with_tools.invoke(trimmed_messages)

        else:  # always do retrieval, calls the retrieval tool
            response = AIMessage(
                content="",
                additional_kwargs={},
                response_metadata={
                    "model": llm.model,
                    "done_reason": "stop",
                    "message": None,
                },
                tool_calls=[
                    {
                        "name": "retrieve",
                        "args": {"query": user_input},
                        "type": "tool_call",
                        "id": str(uuid4()),
                    }
                ],
            )

        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}

    # Step 3: Generate a response using the retrieved content.
    def generate(state: MessagesState):
        """
        Generate answer.
        Args:
            state (MessagesState): State of the conversation.
        Returns:
            MessagesState: State of the conversation with the generated answer.
        """
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]
        # Format into prompt
        retrieved_context = tool_messages[0].content  # retrieved context


        if model_name=="random_urls_model_baseline":
                    #for the random baseline, we render a subset of the urls from the retrieved context
            # parse urls in the answer
            confluence_urls = extract_confluence_urls(retrieved_context)
            # put urls in standard format
            confluence_urls = list(set([
                url_harmonization(url, url_to_standard_url_mapping)
                for url in confluence_urls
            ]))

            all_urls=True
            max_urls=min(3, len(confluence_urls))
            if all_urls:
                size_urls=max_urls
            else:
                size_urls=np.random.randint(0, max_urls+1)

            response_content=" ".join(
                np.random.choice(confluence_urls, size=size_urls, replace=False)
            )
            response = AIMessage(
                content=response_content
            )
        else:
            # end of the prompt to recall important information
            recall_guidelines = """Remember to cite your sources, especially the Confluence URL according to the format I gave you. If you give a document title, give also the URL. And simply say that you don't have the information when you cannot answer. Citing the URL(s) is very important.
            """

            # System message with retrieved context
            system_message_content = f"""You are an assistant working for an internal IT service of a company called {settings.COMPANY_NAME}. Always answer within the context of the company.
            You work as a RAG. I'll give you tutorials about IT services.
            Answer only with the context I give you, detail the steps the user has to follow to solve his issue.
            Do not invent information that is not in the context so if you cannot answer the question,
            say `I don't have this information`.
            When you give an information, always cite from which source (title + document url) your answer comes from. The answer should remain concise.
            So the goal is to provide the detailed steps to solve the issue and to cite the source of the information you provide.
            Your answer must respect the following format:\n
            Sources: Title(s) of the document(s)\n
            URL: url(s) of the relevant document(s)\n
            To solve this issue apply the following steps:\n
            1 Step 1\n
            2 Step 2\n
            3 Step 3\n etc.\n

            \n\nNow here is the context:\n{retrieved_context}\n\n\n
            {recall_guidelines}
            """

        
            if not_citing_source:
                system_message_content = f"""You are an assistant working for an internal IT service of a company called {settings.COMPANY_NAME}. Always answer within the context of the company.
                You work as a RAG. I'll give you tutorials about IT services.
                Answer only with the context I give you, detail the steps the user has to follow to solve his issue.
                Do not invent information that is not in the context so if you cannot answer the question,
                say `I don't have this information`.
                The answer should remain concise.
                So the goal is to provide the detailed steps to solve the issue.
                Your answer must respect the following format:\n
                To solve this issue apply the following steps:\n
                1 Step 1\n
                2 Step 2\n
                3 Step 3\n etc.\n

                \n\nNow here is the context:\n{retrieved_context}\n\n\n
                Remember to simply say that you don't have the information when you cannot answer.
                """


            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]

            # memory management
            # Keep the last 2 messages (user and assistant) in the conversation when testing
            
            trimmed_messages = trimmer.invoke(
                conversation_messages
            )  # Trim to last N messages
            if verbose:
                    print("trimmed messages")
                    for i,message in enumerate(trimmed_messages):
                        #check if the message is a tool call
                        print(f"Message {i}: {message}")
                        print("--------------------------")
                
            prompt = [SystemMessage(system_message_content)] + trimmed_messages
            # call the LLM
            response = llm.invoke(prompt)
        return {"messages": [response]}

    # build the graph
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.set_entry_point("query_or_respond")
    tools = ToolNode([retrieve])
    graph_builder.add_node(tools)
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )  # if the user input needs a retrieval step, call the tool else the llm directly answers
    graph_builder.add_node(generate)
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    memory = MemorySaver()  # memory persistence
    graph = graph_builder.compile(checkpointer=memory)
    return graph


def chat_with_rag(graph, config):
    """Chat with the RAG model.
    Args:
        graph (StateGraph): RAG graph.
        config (dict): Configuration dictionary (not sure what it is yet, sth like {"configurable": {"thread_id": "abc123"}} works).
    """
    print("RAG Chatbot is ready! Type 'exit' to quit.")

    while True:
        input_message = input("\nYou: ")
        if input_message.lower() == "exit":
            break
        input_message = input_message.replace("'", " ").replace("’", " ")
        for step in graph.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="values",
            config=config,
        ):
            if not step["messages"][-1].type == "tool":
                step["messages"][-1].pretty_print()


def get_one_answer_from_rag(graph, config, question):
    """Get one answer from the RAG model.
    Args:
        graph (StateGraph): RAG graph.
        config (dict): Configuration dictionary (not sure what it is yet, sth like {"configurable": {"thread_id": "abc123"}} works).
        question (str): Question to ask the RAG model.
    Returns:
        str: Answer from the RAG model.
    """
    input_message = question
    input_message = input_message.replace("'", " ").replace("’", " ")
    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        pass
    return step["messages"][-1].content


def get_answers_from_rag(graph, config, questions):
    """Get answers from the RAG model.
    Args:
        graph (StateGraph): RAG graph.
        config (dict): Configuration dictionary (not sure what it is yet, sth like {"configurable": {"thread_id": "abc123"}} works).
        questions (list): List of questions to ask the RAG model.
    Returns:
        list: List of answers from the RAG model.
    """
    answers = []
    for question in questions:
        input_message = question
        input_message = input_message.replace("'", " ").replace("’", " ")
        for step in graph.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="values",
            config=config,
        ):
            pass
        answers.append(step["messages"][-1].content)
    return answers


class RAGv3:
    """
    Class to create a RAGv3 model.
    Args:
        vector_base (VectorBase): VectorBase object containing the vectorstore.
        model_name (str): Model name to use for the generation (must be a model in your Ollama models).
        num_predict (int): Number of tokens to predict.
        top_k (int): Number of documents to retrieve.
        threshold (float): Threshold for the relevance score of the retrieved documents.
        always_do_retrieval (bool): Whether to always do retrieval or not.
        verbose (bool): Whether to print the retrieved documents or not.
    """

    def __init__(
        self,
        vector_base: VectorBase,
        model_name="llama3.1:8b",
        num_predict=1000,
        top_k=6,
        threshold=0.1,
        keep_in_mind_last_n_messages=2,
        always_do_retrieval=False,
        verbose=False,
        thread_id="abc123",
        add_external_links_docs=False,
        not_citing_source=False,  # set to True if you want to test the model without citing sources
    ):
        self.vector_base = vector_base
        self.graph = build_graph(
            vector_base=vector_base,
            model_name=model_name,
            num_predict=num_predict,
            top_k=top_k,
            threshold=threshold,
            keep_in_mind_last_n_messages=keep_in_mind_last_n_messages,
            always_do_retrieval=always_do_retrieval,
            verbose=verbose,
            add_external_links_docs=add_external_links_docs,
            not_citing_source=not_citing_source,  # set to True if you want to test the model without citing sources
        )

        self.config = {"configurable": {"thread_id": thread_id}}

    def chat(self):
        chat_with_rag(self.graph, self.config)


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
    ADD_EXTERNAL_LINKS_DOCS = args.add_external_links_docs
    ADD_EXTERNAL_LINKS_DOCS = str_to_bool(ADD_EXTERNAL_LINKS_DOCS)

    TOP_K = args.top_k
    THRESHOLD = args.threshold
    MODEL_NAME = args.model_name
    NUM_PREDICT_TOKENS = args.num_predict_tokens
    KEEP_IN_MIND_LAST_N_MESSAGES = args.keep_in_mind_last_n_messages

    # Create the vectorstore
    vector_base = VectorBase(
        directory=DIRECTORY,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedding_model=EMBEDDING_MODEL,
        nb_chunks=NB_CHUNKS,
    )
    # Create the RAG model
    # RAGv3 is a class that creates a RAG model using langgraph
    RAG = RAGv3(
        vector_base=vector_base,
        model_name=MODEL_NAME,
        num_predict=NUM_PREDICT_TOKENS,
        top_k=TOP_K,
        threshold=THRESHOLD,
        keep_in_mind_last_n_messages=KEEP_IN_MIND_LAST_N_MESSAGES,
        always_do_retrieval=ALWAYS_DO_RETRIEVAL,
        verbose=VERBOSE,
        add_external_links_docs=ADD_EXTERNAL_LINKS_DOCS,
    )

    RAG.chat()
