# 함수 호출을 사용합니다.
# 유저가 시험의 난이도를 커스터마이징 할 수 있도록 하고 LLM이 어려운 문제 또는 쉬운 문제를 생성하도록 합니다.
# 만점이 아닌 경우 유저가 시험을 다시 치를 수 있도록 허용합니다.
# 만점이면 st.ballons를 사용합니다.
# 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 로드합니다.
# st.sidebar를 사용하여 Streamlit app의 코드와 함께 Github 리포지토리에 링크를 넣습니다.

import streamlit as st
from utils import split_file, create_quiz_with_function_calling
from langchain.retrievers import WikipediaRetriever


def grade_score(grade_sheet):
    return sum(value for value in grade_sheet.values())


@st.cache_data(show_spinner="Search Result...")
def use_wiki(topic):
    docs = WikipediaRetriever(top_k_results=3).get_relevant_documents(topic)
    return docs


st.set_page_config(page_title="FullstackGPT Second Assignment", page_icon="🤖")

# initialize session_state
if "grade_sheet" not in st.session_state:
    st.session_state["grade_sheet"] = {
        "1": False,
        "2": False,
        "3": False,
        "4": False,
        "5": False,
    }

docs = None

with st.sidebar:
    st.markdown("https://github.com/Wookji-Yoon/python_challenge_langchain")

    USER_OPENAI_API_KEY = st.text_input(
        label="OpenAI API KEY", placeholder="Fill in your OpenAI API Key"
    )

    if USER_OPENAI_API_KEY:
        st.write("API Key Setting finished!")

        choice = st.selectbox(
            label="Choose what you want to use",
            options=("File", "Wikipedia"),
        )
        if choice == "File":
            file = st.file_uploader(
                label="Upload your text file (only available for .txt)", type="txt"
            )
            if file:
                with st.spinner("Reading File..."):
                    docs = split_file(file)
                st.success("Done")
        if choice == "Wikipedia":
            topic = st.text_input(label="Search keyword in Wikipedia")
            if topic:
                docs = use_wiki(topic)
                st.success("Done!")

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    placeholder = st.empty()
    difficulty = placeholder.selectbox(
        label="Select the difficulty", options=("Easy", "Hard"), index=None
    )
    if difficulty:
        placeholder.empty()
        string_docs = "\n".join(doc.page_content for doc in docs)
        json = create_quiz_with_function_calling(
            api_key=USER_OPENAI_API_KEY, docs=string_docs, difficulty=difficulty
        )

        with st.form("questions_form"):
            for i, question in enumerate(json["questions"]):
                st.write(str(i + 1) + ". " + question["question"])
                value = st.radio(
                    label="Selet an answer",
                    options=[answer["answer"] for answer in question["answers"]],
                    index=None,
                )
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                    st.session_state["grade_sheet"][str(i + 1)] = True
                elif value is not None:
                    st.error("Wrong!")
                    st.session_state["grade_sheet"][str(i + 1)] = False

            st.form_submit_button("Submit")
        score = grade_score(st.session_state["grade_sheet"])
        if score == 5:
            st.balloons()
            st.success("Your Genius!! Your score is 5/5")
        else:
            st.error(f"Your score is {score}/5. Please try again")
