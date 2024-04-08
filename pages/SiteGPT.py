# 함수 호출을 사용합니다.
# 유저가 시험의 난이도를 커스터마이징 할 수 있도록 하고 LLM이 어려운 문제 또는 쉬운 문제를 생성하도록 합니다.
# 만점이 아닌 경우 유저가 시험을 다시 치를 수 있도록 허용합니다.
# 만점이면 st.ballons를 사용합니다.
# 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 로드합니다.
# st.sidebar를 사용하여 Streamlit app의 코드와 함께 Github 리포지토리에 링크를 넣습니다.

import streamlit as st
from utils import (
    load_coudfare_website,
    save_message,
    use_chatmodel,
    define_model_and_memory,
)


st.set_page_config(page_title="FullstackGPT Third Assignment", page_icon="🤖")

if "message" not in st.session_state:
    st.session_state["message"] = [
        {
            "role": "ai",
            "message": "I'm ready to help you",
        }
    ]


USER_OPENAI_API_KEY = None

with st.sidebar:
    st.markdown("https://github.com/Wookji-Yoon/python_challenge_langchain")
    USER_OPENAI_API_KEY = st.text_input(
        label="OpenAI API KEY", placeholder="Fill in your OpenAI API Key"
    )

    if USER_OPENAI_API_KEY:
        st.write("API Key Setting finished!")
        model_and_memory = define_model_and_memory(USER_OPENAI_API_KEY)
        model = model_and_memory["model"]
        memory = model_and_memory["memory"]


if not USER_OPENAI_API_KEY:
    st.markdown(
        """
        # SiteGPT
                
        Ask questions about the content of a Cloudfare.
                
        Start by writing the URL of the website on the sidebar.
        """
    )

if USER_OPENAI_API_KEY:
    retriever = load_coudfare_website(USER_OPENAI_API_KEY)

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
