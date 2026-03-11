import json
import os
import time
import uuid

import streamlit as st
from CIT.config import settings
from utils import get_summary_from_url, replace_urls_with_titles

from CIT.evaluation.utils import load_jsonl, save_jsonl  # type: ignore
from CIT.RAGs.utils import RAGv3, VectorBase, get_one_answer_from_rag  # type: ignore


def show_sources_callback(index_sources):
    st.session_state.source_index = index_sources
def hide_sources_callback():
    st.session_state.source_index = None

def main_app():
    """
    Main application logic goes here.
    """

    # Constants / config
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300

    NUM_PREDICT_TOKENS = 1000
    TOP_K = 6


    # Streamlit app layout
    st.set_page_config(page_title="RAG Chatbot", page_icon="🧠")

    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                width: 500px;
            }
            [data-testid="stSidebar"] > div:first-child {
                width: 350px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    names= st.experimental_user["name"].split(" ")
    first_name=names[0]
    first_name=first_name[0].upper()+first_name[1:]
    st.title(f"Hi {first_name} 👋")
    st.title("💬 Corporate IT Chatbot")
    st.markdown(
        f"RAG based on the [{settings.COMPANY_NAME} Knowledge Base]({settings.CONFLUENCE_BASE_URL}/spaces/CORPORATEITKNOWLEDGEBASE/pages/1311588781/Corporate+IT+-+User+Knowledge+Base). This chatbot is powered by a light open-source model, and was made as part of a master thesis. Therefore it is far from being optimized and is still a beta."
    )  # add the link as an hyperlink
    st.markdown(f"Use the model first for its main purpose: **answering IT-related questions \
                at {settings.COMPANY_NAME}**, based on the [{settings.COMPANY_NAME} Knowledge Base]({settings.CONFLUENCE_BASE_URL}/spaces/CORPORATEITKNOWLEDGEBASE/pages/1311588781/Corporate+IT+-+User+Knowledge+Base). Then, feel free to test\
                 its limits with any questions.\
                 Remember that the model can only answer specific questions if the corresponding information is in the KB.\
                 **Share your feedback using the panel on the left**. \n\n(conversation will be read by humans to improve the service)")


    # Sidebar: Configure RAG parameters
    #st.sidebar.header("🔧 RAG Configuration (click on 'Reinitialize RAG' to make the modifs effective)")
    if "latest_sources" not in st.session_state:
        st.session_state.latest_sources = []
    if "source_index" not in st.session_state:
        st.session_state.source_index = None
    if "sources" not in st.session_state:
        st.session_state.sources = []
    with st.sidebar:
        if st.session_state.source_index is not None:
            st.markdown("### Sources summary")
            print(f"Sources for index {st.session_state.source_index}:")
            print(st.session_state.sources)
            for title,summary in st.session_state.sources[st.session_state.source_index]:
                clean_title=title.strip()
                clean_title=title
                st.markdown(f"**{clean_title}**:\n\n {summary}")
            st.button("Hide sources", key="hide_sources",on_click=hide_sources_callback)#st.session_state.source_index = None
                
        else:
            st.markdown("Sources summary will appear here when you click on the 'Show sources' button after an answer.")

    DIRECTORY = "./src/CIT/documents/run3/confluence_json"
    CHUNK_SIZE=[1500]
    CHUNK_OVERLAP=300
    EMBEDDING_MODEL = "./src/CIT/RAGs/models/all-MiniLM-L6-v2"  # Path to the embedding model
    #EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Huggingface model to use for embeddings
    NB_CHUNKS = -1



  
    MODEL_NAME = "ft_wo0"
    st.session_state.model_name = MODEL_NAME
    NUM_PREDICT_TOKENS = 1000
    TOP_K = 6
    THRESHOLD = 0.1
    KEEP_IN_MIND_LAST_N_MESSAGES = 6
    ADD_EXTERNAL_LINKS_DOCS=False
    ALWAYS_DO_RETRIEVAL = False
    st.session_state.always_do_retrieval = ALWAYS_DO_RETRIEVAL

    USER_PROMPT = ""
    st.session_state.user_prompt = USER_PROMPT
    VERBOSE = False

    # Button to reinitialize the RAG
    if "RAG" not in st.session_state:
        st.session_state.vector_base = VectorBase(
            directory=DIRECTORY,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            embedding_model=EMBEDDING_MODEL,
            nb_chunks=NB_CHUNKS,
        )
        st.session_state.RAG = RAGv3(
            vector_base=st.session_state.vector_base,
            model_name=MODEL_NAME,
            num_predict=NUM_PREDICT_TOKENS,
            user_prompt=st.session_state.user_prompt,
            top_k=TOP_K,
            threshold=THRESHOLD,
            keep_in_mind_last_n_messages=KEEP_IN_MIND_LAST_N_MESSAGES,
            always_do_retrieval=st.session_state.always_do_retrieval,
            verbose=VERBOSE,
            add_external_links_docs=ADD_EXTERNAL_LINKS_DOCS,
        )

    def chat_stream(prompt,graph,config):
        response = get_one_answer_from_rag(
            graph=graph,
            config=config,
            question=prompt,
        )
        return response


    def save_feedback(index):
        st.session_state.chat_history[index]["feedback"] = st.session_state[
            f"feedback_{index}"
        ]
        print("feedback given at index", index)

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_input = st.chat_input("Ask a question...")

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                # feedback = message.get("feedback", None)
                # st.session_state[f"feedback_{i}"] = feedback
                st.feedback(
                "stars",
                key=f"feedback_{i}",
                on_change=save_feedback,
                args=[i],
            )
                index_sources=(i-1)//2#because no sources stored for the user
                if len(st.session_state.sources[index_sources])>0:
                    st.button("Show sources (left panel)", key=f"sources_{i}",on_click=show_sources_callback,args=(index_sources,))
                       # st.session_state.source_index = index_sources
    # Handle new question
    if user_input:
        # Show user message
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        if not st.session_state.RAG:
            st.error("Please reinitialize the RAG first using the sidebar.")

        # Get RAG response
        with st.chat_message("assistant"):
            response = ""
            placeholder = st.empty()
            for char in chat_stream(user_input,graph=st.session_state.RAG.graph,config=st.session_state.RAG.config):
                time.sleep(0.001)  # Simulate streaming response
                response += char
                placeholder.markdown(response)
            response,urls,titles=replace_urls_with_titles(response,st.session_state.RAG.mapping_urls_titles)
            placeholder.markdown(response)
            st.session_state.latest_sources = [(titles[i],
                                                get_summary_from_url(url, st.session_state.RAG.mapping_urls_paths))
                                                for i,url in enumerate(urls) 
                                                if url in st.session_state.RAG.mapping_urls_paths]
            st.session_state.sources.append(st.session_state.latest_sources)


            st.session_state.chat_history.append({"role": "assistant", "content": response})
            i=len(st.session_state.chat_history)-1
            st.feedback(
                "stars",
                key=f"feedback_{i}",
                on_change=save_feedback,
                args=[i],
            )
            if len(urls)>0:
                st.button("Show sources (left panel)", key=f"sources_{i}",on_click=show_sources_callback,args=((i-1)//2,))

            #save the messages to a jsonl file
            path_user_messages = f"./src/CIT/UI/messages/{st.session_state.username}.jsonl"
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            new_messages = [{"role": "user", "content": user_input, "timestamp": current_time},
                            {"role": "assistant", "content": response, "timestamp": current_time}]
            if not os.path.exists(path_user_messages):
                save_jsonl(path_user_messages, new_messages)
            else:
                last_messages=load_jsonl(path_user_messages)
                last_messages=last_messages+new_messages
                save_jsonl(path_user_messages, last_messages)

    st.markdown("---")

    # Sidebar: Feedback section
    st.sidebar.header("📝 General Feedback")
    with st.sidebar:
        st.radio("Do you expect the model to speak like a human ? Or it does not matter as long as its answers help you ?",
                 options=["It does not matter", "I expect the model to speak like a human"],
                 index=0,
                 key="human_speaking")
        st.slider(
            "How useful is the model, i.e answers your question",
            value=3,
            min_value=0,
            max_value=5,
            key="usefulness",
        )
        st.radio(
            "Would you use the model again ?", options=["Yes", "No"], index=0, key="use_again"
        )
        st.radio(
            "Would you have easily found the answers in the knowledge base ?", options=["Yes", "No"], index=0, key="answers_in_kb"
        )
        
        st.text_area("Provide your feedback", value="", height=100, key="feedback_text",placeholder="Please provide your feedback here.")
        st.button("Submit feedback", key="submit_feedback")
        # save feedback to a file
        if st.session_state["submit_feedback"]:
            feedback = {
                "username": st.session_state.username,
                "model_name": st.session_state.model_name,
                "usefulness": st.session_state["usefulness"],
                "use_again": st.session_state["use_again"],
                "human_speaking": st.session_state["human_speaking"],
                "answers_in_kb": st.session_state["answers_in_kb"],
                "feedback_text": st.session_state["feedback_text"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }
            # add the rated answers:
            feedback["rated_answers"] = []
            print(st.session_state.chat_history)
            for i in range(len(st.session_state.chat_history)):
                if st.session_state.chat_history[i]["role"] == "assistant" and "feedback" in st.session_state.chat_history[i]:
                    question = st.session_state.chat_history[i - 1]["content"]
                    answer = st.session_state.chat_history[i]["content"]
                    rating = st.session_state.chat_history[i]["feedback"]
                    feedback["rated_answers"].append(
                        {
                            "question": question,
                            "answer": answer,
                            "rating": rating,
                        }
                    )


            # Generate a unique ID for the feedback
            feedback["id"] = str(uuid.uuid4())
            with open(f"./src/CIT/UI/feedbacks/feedback_{feedback['id']}.json", "a") as f:
                json.dump(feedback, f)
            st.success("Feedback submitted!")

def login_screen():
    st.header("This app is private.")
    st.subheader("Please log in with your Acme Corp account")
    st.button("Log in with Microsoft", on_click=st.login)
if not st.experimental_user.is_logged_in:
    login_screen()
else:
    st.session_state.username = st.experimental_user["name"]
    #st.write(st.experimental_user)
    main_app()
    st.button("Log out", on_click=st.logout)
