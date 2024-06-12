import json
import sys
import os
import boto3

# We will be using titan embedding model for creating vectors to generate embedding.

from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Data Ingistion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader


# Vector embedding and vector stores

from langchain.vectorstores import FAISS


#LLModule

from langchain.prompts import PromptTemplate
from langchain.chains import retrieval_qa

# Bedrock Client
bedrock = boto3.client(service_name = "bedrock-runtime", region_name = "us-east-1")


#Get embeddings model from bedrock
bedrock_embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client= bedrock)


# Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader('data')
    documents = loader.load
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunks_overlap = 200)

    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embedding and vector stores

def get_vector_store(docs):
        vectorstore_faiss =  FAISS.from_documents(
        docs,
        bedrock_embedding
        )

        vectorstore_faiss.save_local("faiss_index")

def get_llm():
      llm = Bedrock(model_id = "mistral.mistral-7b-instruct-v0:2", client = bedrock)
      return llm
 