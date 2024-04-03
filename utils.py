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
def embed_file(file):
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

    embeddings = OpenAIEmbeddings()
    cache_dir = LocalFileStore("./streamlit_cache/embeddings/{file.name}")

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(split_docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever


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
