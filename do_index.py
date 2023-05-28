# Configure if CUDA should be enabled, see TROUBLESHOOTING for details on installing CUDA enabled libraries
CUDA = False

import profile
p = profile.Profile()
p.log("Iniciando ")

import os
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
p.log("Main langchain ")

from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
p.log("Langchain Embeddings ")

loader = DirectoryLoader('./new_papers/new_papers/', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
p.log(f"Number of pdf documents: {len(documents)} ")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
p.log(f"Chunks of texts: {len(texts)} ")

from langchain.embeddings import HuggingFaceInstructEmbeddings
if CUDA :
	instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"})
	p.log("Initializing embeddings with CUDA ")
else :
	instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
	p.log("Initializing embeddings without CUDA ")

embedding = instructor_embeddings
persist_directory = 'db'
vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=persist_directory)
vectordb.persist()
p.log("Indexing complete ")
