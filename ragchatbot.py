import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import logging

# Load environment variables
load_dotenv()

# Configure logging to store the document fetching process
logging.basicConfig(filename="doc_fetching_logs.log", level=logging.INFO)

# Load the GROQ and OpenAI API keys
groq_api_key = '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'
os.environ["GOOGLE_API_KEY"] = "AIzaSyDLggVsJM9ukyY_XmPpRu7_j28IVnr24kU"

st.title("RAG-Chatbot")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Mixtral-8x7b-32768")

prompt = ChatPromptTemplate.from_template(
    """
    You are an expert document analyst with deep contextual understanding. 
    Follow these guidelines when answering:
    - Provide precise, relevant answers directly from the source documents
    - If the answer is not definitively in the context, clearly state that
    - Prioritize accuracy over speculation
    - Include source references or page numbers if possible
    - Break down complex answers into clear, structured explanations
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./pdf")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len, is_separator_regex=False)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings

# Initialize vector store DB once
if "vectors" not in st.session_state:
    vector_embedding()

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Enter"):
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        end = time.process_time()
        
        st.write("Response time: {:.2f} seconds".format(end - start))
        st.write(response['answer'])
        
        # Log the document parts fetched
        for doc in response["context"]:
            logging.info(f"Fetched Document: {doc.page_content}")
    else:
        st.error("Vector store DB is not initialized.")