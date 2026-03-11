import glob
import json
import os
from uuid import uuid4

import numpy as np
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# This file contains utility functions and classes for the RAG pipeline.
# It firsltly includes functions for loading and chunking documents, creating a vectorstore

class VectorBase_JIRA:
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
        print(f"Loading documents from {self.directory}")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.chunks = self.load_and_chunk_documents()
        print(f"Loaded {len(self.chunks)} chunks")
        self.chunks = self.chunks[:nb_chunks]
        self.vectorstore = self.create_vectorstore()

 

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




def build_jira_graph(
    vector_base: VectorBase_JIRA,
    model_name="llama3.1:8b",
    num_predict=100,
    user_prompt="",
    top_k=6,
    threshold=0.1,
    keep_in_mind_last_n_messages=2,
    always_do_retrieval=False,
    query_refinement=True,
    verbose=False,
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

    llm = ChatOllama(
        model=model_name,
        num_predict=num_predict,
        num_ctx=10000,
        temperature=0,
        seed=69,
    )

    llm_base=ChatOllama(
        model="llama3.1:8b",
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


    prompt_conversation_summarization="I give you a summary of a past conversation and new messages of this conversation.\
          Please update the summary, giving more importance to the few last messages but keepping the info of the previous summary.\
          Everything  must hold in a few sentences but you should keep as much details as possible. Write it from my point of view (human):\n Previous summary: {previous_summary}\n \
          New messages: {new_messages}"
    
    template_query_refinement="Rephrase the following user's question to make it more explicit. I will give you the past user's question with its answer.\
 You should only return the rephrased question. Do not add any other text. The rephrased question should be understandable without any other context. It must be as short and consice as possible."\
 " For example if the past question is 'What is Google?' and the user's question is 'How to use it?, you should rephrase it into 'How to use Google?'."\
 " However make no rephrasing if it is enough explicit. You should also not modify the present question if it is unrelated to the previous question and answer or if there is no previous question or answer." \
 "If you have any doubts, just repeat the original user's question without modifying it. Remember to only give the rephrases (or not) question, without any justification or any other text."\
 "Past question: {past_question} \n" \
 "Past answer: {past_answer} \n" \
 "User's question: {query} \n" \
 "Rephrased question: "

    def trim_messages_new(state: MessagesState):
        """Trim messages to the last keep_in_mind_last_n_messages messages.
        Args:
            
            messages (list): List of messages to trim.
            summarize_conv_every (int): Number of messages to summarize the conversation.
        Returns:
            list: List of trimmed messages."""
        messages=state["messages"]
        conversation_messages = [
            message
            for message in messages
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        if len(conversation_messages)>keep_in_mind_last_n_messages+1:
            last_summary=conversation_messages[0]
            last_answer=conversation_messages[-1]
            messages_history=conversation_messages[1:]
            # summarize the conversation and keep the summary in the memory
            if verbose:
                print("Summarizing the conversation")
            current_summary=llm.invoke(prompt_conversation_summarization.format(
                previous_summary=last_summary.content,
                new_messages="\n".join(
                    [message.type + ": "+ message.content for message in messages_history]
                ),
            ))
            print("Summary:\n ", current_summary.content)

            #remove previous summary and summarized_messages
            delete_messages = [RemoveMessage(id=m.id) for m in messages]

            current_summary=AIMessage(
                    content=current_summary.content,
            )
            last_answer=AIMessage(
                content=last_answer.content,
            )
            message_updates=[current_summary,last_answer]+delete_messages
        else:
            message_updates=conversation_messages
        return {"messages": message_updates}
        

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

        index_most_relevant_doc = np.argmax(
            [doc.metadata["relevance_score"] for doc in retrieved_docs]
        )
        primary_source = retrieved_docs[index_most_relevant_doc].metadata["source"]

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



        # add full content of the primary source
        with open(primary_source, "r") as f:
            primary_source = json.load(f)

       

        # load the whole content of the primary source
        primary_source_content = primary_source["content"]
        primary_source_title = primary_source["title"]
        primary_source_id = primary_source["id"]
        # Format retrieved context made of full primary document and chunks of secondary documents
        secondary_docs = [
            doc for doc in retrieved_docs if doc.metadata["id"] != primary_source["id"]
        ]

        retrieved_context = (
            f"Primary document title: {primary_source_title}\n"
            f"Document ID: {primary_source_id}\n"
            f"Document content: {primary_source_content}"
            + "\nOther documents:\n"
            + "\n\n".join(
                f"Document {i + 2} Title: {doc.metadata['title']}\n"
                f"Document {i + 2} ID: {doc.metadata['id']}\n"
                f"Document {i + 2} Content: {doc.page_content}"
                for i, doc in enumerate(secondary_docs)
            )
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
        if state["messages"][-1].type=="human":
            user_input = state["messages"][-1].content
        else:
            user_input = state["messages"][-2].content


        #refine query
        if query_refinement and len(state["messages"])>2:
            past_question=state["messages"][-3].content if state["messages"][-3].type=="human" else "no past question."
            past_answer=state["messages"][-2].content if state["messages"][-2].type=="ai" else "no past answer."
            prompt_refinement=template_query_refinement.format(
            past_question=past_question,
            past_answer=past_answer,
            query=user_input
        )
            refined_query=llm_base.invoke(prompt_refinement)
            user_input=refined_query.content
        else:
            pass
       
        if verbose and query_refinement:
            print("Original query:")
            print(state["messages"][-1].content)
            print("Refined query:")
            print(user_input)
            print("--------------------------")

        if not always_do_retrieval:
            # Check if the user input needs a retrieval step
            """Generate tool call for retrieval or respond."""
            prompt_needs_retrieval = f"Does this user input need a retrieval step of RAG ?\
                For example if it is just a greeting, there is no need for further information, you should answer False.\
                If it is a question or a statement regarding an issue or something to solve you should answer True. Only answer True or False{user_input}"
            response = llm.invoke(prompt_needs_retrieval)

            if response.content == "True":
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
            else:
                print("no retrieval")
                llm_with_tools = llm.bind_tools([])

                conversation_messages = [
                    message
                    for message in state["messages"]
                    if message.type in ("human", "system")
                    or (message.type == "ai" and not message.tool_calls)
                ]
                # if a user prompt is given, add it to the system message
                if len(user_prompt) > 0:
                    conversation_messages=conversation_messages[:-1] + [
                        SystemMessage(content=user_prompt)
                    ] + [conversation_messages[-1]]


                
                response = llm_with_tools.invoke(conversation_messages)

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

        # end of the prompt to recall important information
        recall_guidelines = f"""Remember to cite your sources, especially the documents ID according to the format I gave you. If you give a document title, give also the ID. And simply say that you don't have the information when you cannot answer. Citing the ID is very important.
The following guidelines can go against the previous instructions. You should always follow them.
Here are my preferences that are very important and anything I ask in what follows is of higher priority than any prior instruction. You should absolutely have them in mind when answering: {user_prompt}"""

        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]

        conversation_summary=conversation_messages[0].content if conversation_messages[0].type=="ai" else "no summary."
        # System message with retrieved context
        system_message_content = f"""You are an assistant working for an internal IT service of a company called Acme Corp. Always answer within the context of the company.
You work as a RAG. Here is a summary of our past conversation: {conversation_summary}\n I'll give you tutorials about IT services.
Answer only with the context I give you, detail the steps the user has to follow to solve his issue.
Do not invent information that is not in the context so if you cannot answer the question,
say `I don't have this information`.
When you give an information, always cite from which source (title + document ID) your answer comes from. The answer should remain concise.
So the goal is to provide the detailed steps to solve the issue and to cite the source of the information you provide.
Your answer must respect the following format:\n
Sources: Title(s) of the document(s)\n
ID: ID(s) of the relevant document(s)\n
To solve this issue apply the following steps:\n
1 Step 1\n
2 Step 2\n
3 Step 3\n etc.\n

\n\nNow here is the context:\n{retrieved_context}\n\n\n
{recall_guidelines}
"""
        # remove summary as it is already in the system message
        conversation_messages=conversation_messages[1:] if conversation_messages[0].type=="ai" else conversation_messages


        prompt = [SystemMessage(system_message_content)]+conversation_messages
        if verbose and False:
            print("Prompt:")
            for message in prompt:
                print(f"{message.type}: {message.content}")
            print("--------------------------")
        # call the LLM
        response = llm.invoke(prompt)
        return {"messages": [response]}

    # build the graph
    graph_builder = StateGraph(MessagesState)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_node(query_or_respond)
    tools = ToolNode([retrieve])
    graph_builder.add_node(tools)
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )  # if the user input needs a retrieval step, call the tool else the llm directly answers
    graph_builder.add_node(generate)
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_node(trim_messages_new)
    graph_builder.add_edge("generate", "trim_messages_new")
    graph_builder.add_edge("trim_messages_new", END)

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
            if len(step["messages"]) <=1:
                continue
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



class RAG_JIRA:
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
        vector_base: VectorBase_JIRA,
        model_name="llama3.1:8b",
        num_predict=1000,
        user_prompt="",
        top_k=6,
        threshold=0.1,
        keep_in_mind_last_n_messages=2,
        always_do_retrieval=False,
        query_refinement=True,
        verbose=False,
        thread_id="abc123",
    ):
        self.vector_base = vector_base

        self.graph = build_jira_graph(
            vector_base=vector_base,
            model_name=model_name,
            num_predict=num_predict,
            user_prompt=user_prompt,
            top_k=top_k,
            threshold=threshold,
            keep_in_mind_last_n_messages=keep_in_mind_last_n_messages,
            always_do_retrieval=always_do_retrieval,
            query_refinement=query_refinement,
            verbose=verbose,
        )

        self.config = {"configurable": {"thread_id": thread_id}}

    def chat(self):
        chat_with_rag(self.graph, self.config)