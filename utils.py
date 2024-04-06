import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.globals import set_debug
from langchain.memory import ConversationBufferMemory

set_debug(True)


@st.cache_resource()
def define_model_and_memory(openai_key):

    model = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        api_key=openai_key,
    )
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
    )
    return {"model": model, "memory": memory}


def invoke_chain_and_save_memory(memory, chain, question):
    result = chain.invoke(question)
    memory.save_context(
        {"input": question},
        {"output": result.content},
    )
    return result.content


from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import FAISS


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, api_key):
    file_content = file.read()
    file_path = f"./streamlit_cache/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    loader = UnstructuredFileLoader(file_path)
    load_docs = loader.load()

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=300,
        chunk_overlap=50,
    )

    split_docs = splitter.split_documents(documents=load_docs)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    cache_dir = LocalFileStore("./streamlit_cache/embeddings/{file.name}")

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(split_docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever


@st.cache_data(show_spinner="Reading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./streamlit_cache/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    loader = UnstructuredFileLoader(file_path)
    load_docs = loader.load()

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=500,
        chunk_overlap=0,
    )

    split_docs = splitter.split_documents(documents=load_docs)

    return split_docs


def save_message(role, message):
    st.session_state["message"].append(
        {
            "role": role,
            "message": message,
        }
    )


from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


from langchain.schema.runnable import RunnableLambda, RunnablePassthrough


def use_chatmodel(question, retriever, model, memory):
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
            Use the following reference document to answer the question of human. \n
            If you don't know the answer, just say that you don't know. Don't try to make up an answer.\n
            \n
            [reference document]\n
            {context}\n\n
            """
            ),
            SystemMessagePromptTemplate.from_template(
                """
            Below is the chat history\n
            {memory}\n\n
            """
            ),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    def retrieve_docs(question):
        docs = retriever.invoke(question)
        return "\n".join(doc.page_content for doc in docs)

    def load_memory(_):
        return memory.load_memory_variables({})["chat_history"]

    final_chain = (
        {
            "context": RunnableLambda(retrieve_docs),
            "question": RunnablePassthrough(),
            "memory": RunnableLambda(load_memory),
        }
        | prompt
        | model
    )

    result = invoke_chain_and_save_memory(
        chain=final_chain, question=question, memory=memory
    )
    return result


@st.cache_data(show_spinner="Create Quiz...")
def create_quiz_with_function_calling(api_key, docs, difficulty):
    import json

    function = {
        "name": "create_quiz",
        "description": "function that takes a list of questions and answers and returns a quiz",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                            },
                            "answers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "answer": {
                                            "type": "string",
                                        },
                                        "correct": {
                                            "type": "boolean",
                                        },
                                    },
                                    "required": ["answer", "correct"],
                                },
                            },
                        },
                        "required": ["question", "answers"],
                    },
                }
            },
            "required": ["questions"],
        },
    }

    llm = ChatOpenAI(temperature=0.1, api_key=api_key).bind(
        function_call={"name": "create_quiz"},
        functions=[function],
    )

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
            You are a helpful assistant that is role playing as a teacher.        
            Based ONLY on the following context make 5 (FIVE) questions to test the user's knowledge about the text.    
            Each question should have 4 answers, three of them must be incorrect and one should be correct.
            Please answer as the json format.
            
            When the diffuculty is 'Hard', answer options should be sentence.
            When the difficulty is 'Easy', answer options shoud be word.
         
            Here are the Question examples(use (o) to signal the correct answer):
         
            Question: What is the color of the ocean?
            Answers(Easy): Red|Yellow|Green|Blue(o)
            Answers(Hard): The coror is red | The color is yellow | The color is green | The color is blue
            
            Question: What is the capital of Georgia?
            Answers(Easy): Baku|Tbilisi(o)|Manila|Beirut
            Answers(Hard): The capital of Georgia is Baku | The capital of Georgia is Tbilisi | The capital of Georgia is Manila | The capital of Georgia is Beirut
            \n
            NOW, IT's YOUR TURN!
            CONTEXT IS BELOW 
            --------
            {context}\n\n
            """
            ),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"context": docs})

    response = response.additional_kwargs["function_call"]["arguments"]

    result = json.loads(response)

    return result
