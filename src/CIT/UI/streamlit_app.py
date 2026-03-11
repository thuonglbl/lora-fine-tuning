#DEV UI for the RAG chatbot based on the Corporate IT Knowledge Base

import json
import os
import time
import uuid

import streamlit as st
from CIT.config import settings
from utils import get_summary_from_url, replace_urls_with_titles  # type: ignore

from CIT.evaluation.utils import load_jsonl, save_jsonl  # type: ignore
from CIT.RAGs.utils import RAGv3, VectorBase, get_one_answer_from_rag  # type: ignore

path_credentials="./src/CIT/UI/auth/credentials.json"
with open(path_credentials, "r") as f:
    USER_CREDENTIALS = json.load(f)


for key in ["logged_in", "username", "show_reset"]:
    if key not in st.session_state:
        st.session_state[key] = False if key == "logged_in" else ""

def login():
    st.title("Login")

    username = st.text_input("email")
    password = st.text_input("Password", type="password")
    

    if st.button("Login"):
        if True:
            st.session_state.logged_in = True
            st.session_state.username = "Michel"
        elif username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
        else:
            st.error("Invalid email or password")

    if st.button("Forgot Password?"):
        st.session_state.show_reset = True

def reset_password():
    st.title("Reset Password")

    username = st.text_input("email for reset")
    new_password = st.text_input("New Password", type="password", key="new_pass")
    confirm_password = st.text_input("Confirm Password", type="password", key="confirm_pass")

    if st.button("Submit"):
        if username not in USER_CREDENTIALS:
            st.error("email not found.")
        elif new_password != confirm_password:
            st.error("Passwords do not match.")
        elif not new_password:
            st.error("Password cannot be empty.")
        else:
            USER_CREDENTIALS[username] = new_password
            with open(path_credentials, "w") as f:
                json.dump(USER_CREDENTIALS, f)
            st.success("Password updated successfully. Please log in.")
            st.session_state.show_reset = False

    if st.button("Back to Login"):
        st.session_state.show_reset = False


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
    st.set_page_config(page_title="RAG Chatbot (dev)", page_icon="🧠")

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
    first_name = st.session_state.username.split(".")[0]
    first_name=first_name[0].upper()+first_name[1:]
    st.title(f"Hi {first_name} 👋")
    st.title("💬 Corporate IT Chatbot (DEV)")
    st.markdown(
        f"RAG based on the [{settings.COMPANY_NAME} Knowledge Base]({settings.CONFLUENCE_BASE_URL}/spaces/CORPORATEITKNOWLEDGEBASE/pages/1311588781/Corporate+IT+-+User+Knowledge+Base)"
    )  # add the link as an hyperlink
    st.markdown(f"Use the model first for its main purpose: **answering IT-related questions \
                at {settings.COMPANY_NAME}**, based on the [{settings.COMPANY_NAME} Knowledge Base]({settings.CONFLUENCE_BASE_URL}/spaces/CORPORATEITKNOWLEDGEBASE/pages/1311588781/Corporate+IT+-+User+Knowledge+Base). Then, feel free to test\
                 its limits with any questions.\
                 Remember that the model can only answer specific questions if the corresponding information is in the KB.\
                 **Share your feedback using the panel on the left**.")


    # Sidebar: Configure RAG parameters
    st.sidebar.header("🔧 RAG Configuration (click on 'Reinitialize RAG' to make the modifs effective)")
    if "latest_sources" not in st.session_state:
        st.session_state.latest_sources = []
    if "source_index" not in st.session_state:
        st.session_state.source_index = None
    if "sources" not in st.session_state:
        st.session_state.sources = []
    with st.sidebar:
        st.markdown("### Sources")
        if st.session_state.source_index is not None:
            for title,summary in st.session_state.sources[st.session_state.source_index]:
                clean_title=title.strip()
                clean_title=title
                st.markdown(f"**{clean_title}**:\n\n {summary}")
        else:
            st.markdown("_No sources to display._")

    st.sidebar.selectbox(
        "Directory", options=["last","updated","old","altered"], index=0, key="directory"
    )
    map_directory_path = {
        "old": "./src/CIT/documents/confluence_json_without_root_with_titles",
        "updated": "./src/CIT/documents/run2/confluence_json",
        "altered": "./src/CIT/documents/run2/altered",
        "last": "./src/CIT/documents/run3/confluence_json",
    }


    CHUNK_SIZE = st.sidebar.slider(
        "Chunk size",
        value=CHUNK_SIZE,
        min_value=100,
        step=50,
        max_value=3000,
        key="chunk_size",
    )
    CHUNK_OVERLAP = st.sidebar.slider(
        "Chunk overlap",
        value=CHUNK_OVERLAP,
        min_value=0,
        max_value=1000,
        step=25,
        key="chunk_overlap",
    )
    st.sidebar.selectbox(
        "Embedding model",
        options=["sentence-transformers/all-MiniLM-L6-v2"],
        index=0,
        key="embedding_model",
    )
    st.sidebar.selectbox(
        "Model name",
        options=["user_ft0505_sd_urls","ft_wo0", "llama3.1:8b","llama3.1:8b-instruct-q4_K_M","user_ft_22.5_1epoch"],
        index=0,
        key="model_name",
    )
    NUM_PREDICT_TOKENS = st.sidebar.slider(
        "Max num predict tokens",
        value=NUM_PREDICT_TOKENS,
        min_value=1,
        max_value=2000,
        step=20,
        key="num_predict_tokens",
    )
    st.sidebar.slider(
        "Number of previous messages the model should keep in mind.",
        value=8,
        min_value=1,
        max_value=20,
        step=1,
        key="keep_in_mind_n_messages",
    )

    TOP_K = st.sidebar.slider(
        "Top K", value=TOP_K, min_value=1, max_value=30, step=1, key="top_k"
    )
    st.sidebar.slider(
        "Threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key="threshold"
    )
    st.sidebar.checkbox(
        "Always do retrieval", value=False, key="always_do_retrieval"
    )
    st.sidebar.checkbox("Verbose (for debugging)", value=True, key="verbose")

    st.sidebar.checkbox(
        "Add external links docs to retrieved context", value=False, key="add_external_links_docs"
    )

    st.sidebar.text_area(
        "User prompt",
        value="",
        placeholder="Enter your overall preferences here if you feel the model should behave a certain way.",
        height=100,
        key="user_prompt",
    )
    # Button to reinitialize the RAG
    if "RAG" not in st.session_state:
        vector_base = VectorBase(
            directory=map_directory_path[st.session_state.directory],
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
            embedding_model=st.session_state.embedding_model,
            nb_chunks=-1,
        )
        st.session_state.RAG = RAGv3(
            vector_base=vector_base,
            model_name=st.session_state.model_name,
            num_predict=st.session_state.num_predict_tokens,
            user_prompt=st.session_state.user_prompt,
            top_k=st.session_state.top_k,
            threshold=st.session_state.threshold,
            keep_in_mind_last_n_messages=st.session_state.keep_in_mind_n_messages,
            always_do_retrieval=st.session_state.always_do_retrieval,
            add_external_links_docs=st.session_state.add_external_links_docs,
            verbose=st.session_state.verbose,
        )

    if st.sidebar.button("🔄 Reinitialize RAG"):
        with st.spinner("Reinitializing RAG..."):
            vector_base = VectorBase(
                directory=map_directory_path[st.session_state.directory],
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
                embedding_model=st.session_state.embedding_model,
                nb_chunks=-1,
            )
            st.session_state.RAG = RAGv3(
                vector_base=vector_base,
                model_name=st.session_state.model_name,
                num_predict=st.session_state.num_predict_tokens,
                user_prompt=st.session_state.user_prompt,
                top_k=st.session_state.top_k,
                threshold=st.session_state.threshold,
                keep_in_mind_last_n_messages=st.session_state.keep_in_mind_n_messages,
                always_do_retrieval=st.session_state.always_do_retrieval,
                add_external_links_docs=st.session_state.add_external_links_docs,
                verbose=st.session_state.verbose,
            )
            st.success("RAG reinitialized!")


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
                    if st.button("Show sources (left panel)", key=f"sources_{i}"):
                        st.session_state.source_index = index_sources
                    
                
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
                if st.button("Show sources (left panel)", key=f"sources_{i}"):
                        st.session_state.source_index = (i-1)//2

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
        st.radio("Do you expect the model to speak like a human ? Or it does not matter as long as its answers helps you ?",
                 options=["it does not matter", "i expect the model to speak like a human"],
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
            "Would you use the model again ?", options=["yes", "no"], index=1, key="use_again"
        )
        st.radio(
            "Would you have easily found the answers in the knowledge base ?", options=["yes", "no"], index=1, key="answers_in_kb"
        )
        st.text_area("provide your feedback", value="", height=100, key="feedback_text")
        st.button("Submit feedback", key="submit_feedback")
        # save feedback to a file
        if st.session_state["submit_feedback"]:
            feedback = {
                "username": st.session_state["username"],
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


# def login_screen():
#     st.header("This app is private.")
#     st.subheader("Please log in.")
#     st.button("Log in with Microsoft", on_click=st.login)
# if not st.experimental_user.is_logged_in:
#     login_screen()
# else:
#     main_app()
#     st.button("Log out", on_click=st.logout)



# Routing
if st.session_state.logged_in:
    main_app()
elif st.session_state.show_reset:
    reset_password()
else:
    login()