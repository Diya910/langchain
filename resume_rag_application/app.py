from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Resume RAG App")
st.title("📝 Resume Q&A")

upload_file = st.file_uploader("Upload your resume", type=["pdf"])
query = st.text_input("Ask a question about your resume:")

if upload_file:
    with open("resume.pdf", "wb") as f:
        f.write(upload_file.read())
    loader = PyPDFLoader("resume.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150,chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="resume_rag_app")
    vectorstore.persist()

    retriever = vectorstore.as_retriever()
    chatbot = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain(llm=chatbot, retriever=retriever, memory=memory)

    if query:
        with st.spinner("Thinking..."):
            response = chain.run({"question": query})
            st.write(response)
        st.success("Question processed successfully!")
    
