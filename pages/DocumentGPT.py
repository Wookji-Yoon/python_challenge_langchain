# Migrate the RAG pipeline you implemented in the previous assignments to Streamlit.
# Implement file upload and chat history.
# Allow the user to use its own OpenAI API Key, load it from an st.input inside of st.sidebar
# Using st.sidebar put a link to the Github repo with the code of your Streamlit app.


import streamlit as st
from utils import embed_file, save_message, use_chatmodel, define_model_and_memory

st.set_page_config(page_title="FullstackGPT First Assignment", page_icon="🤖")

# initialize session_state
if "message" not in st.session_state:
    st.session_state["message"] = [
        {
            "role": "ai",
            "message": "I'm ready to help you",
        }
    ]

file = False

with st.sidebar:
    USER_OPENAI_API_KEY = st.text_input(
        label="OpenAI API KEY", placeholder="Fill in your OpenAI API Key"
    )
    if USER_OPENAI_API_KEY:
        st.write("API Key Setting finished!")
        model_and_memory = define_model_and_memory(USER_OPENAI_API_KEY)
        model = model_and_memory["model"]
        memory = model_and_memory["memory"]

        file = st.file_uploader(
            label="Upload your text file (only available for .txt)", type="txt"
        )


if file:
    retriever = embed_file(file, USER_OPENAI_API_KEY)

    for message in st.session_state["message"]:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

    question = st.chat_input("Enter your message")
    if question:
        save_message("human", question)
        with st.chat_message("human"):
            st.markdown(question)
        answer = use_chatmodel(question, retriever, model, memory)
        if answer:
            save_message("ai", answer)
            with st.chat_message("ai"):
                st.markdown(answer)

    else:
        st.session_state["messages"] = [
            {
                "role": "ai",
                "message": "I'm ready to help you",
            }
        ]
