import boto3
import streamlit as st
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

#Bedrock client
bedrock = boto3.client(service_name = "bedrock-runtime", region_name = "us-east-1")

#Get embeddings model from bedrock
bedrock_embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client= bedrock)


def get_document():
    loader = PyPDFDirectoryLoader('data')
    documents = loader.load
    text_splitter = RecursiveCharacterTextSplitter(
                                               chunk_size =1000 , 
                                               chunk_overlap =500)
    docs = text_splitter.split_documents(documents)
    return docs
    # len(docs)

