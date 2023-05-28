# Parte do codigo que cria as embeddings sem Openai: https://colab.research.google.com/drive/17eByD88swEphf-1fvNOjf_C79k0h2DgF?usp=sharing#scrollTo=wKfX4vX-5RFT
# Parte que ao inves de usar a chain da OpenAI, usa instructor-xl: https://colab.research.google.com/drive/1oCrSkij1NNedV_yZzRTTv0VUjAAxSgCB?usp=sharing

import profile
p = profile.Profile()
p.log("Iniciando ")
import os
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
p.log("Main langchain ")
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
p.log("Langchain Embeddings ")
#print("Carregando documentos PDF")
#loader = DirectoryLoader('./new_papers/new_papers/', glob="./*.pdf", loader_cls=PyPDFLoader)
#documents = loader.load()
#print(f"Numero de documentos lidos: {len(documents)}")

#print("splitting the text into chunks")
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#texts = text_splitter.split_documents(documents)
#print(f"Total de chunks: {len(texts)}")

from langchain.embeddings import HuggingFaceInstructEmbeddings
#Linhas a seguir utiliza CUDA, normalmente suporte apenas para NVIDIA
#print("Inicializando Embeddings com cuda")
#instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"})
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
p.log("Inicializando Embeddings sem cuda ")

embedding = instructor_embeddings
# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

#vectordb = Chroma.from_documents(documents=texts, 
#                                 embedding=embedding,
#                                 persist_directory=persist_directory)
#vectordb.persist()
#vectordb = None

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)
p.log("Carregando vectordb ")
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
retriever.search_kwargs

from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import torch
import transformers
p.log("Transformers and torch ")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
p.log("Tokenizer ")

model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
p.log("Model loaded ")

pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=256
)
local_llm = HuggingFacePipeline(pipeline=pipe)
qa_chain = RetrievalQA.from_chain_type(llm=local_llm, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)
p.log("Chain created ")

def wrap_text_preserve_newlines(text, width=110):
    return text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

def answer(query, qa_chain):
    print(f"\n\n---\n{query}")
    process_llm_response(qa_chain(query))
    
answer('who are the authors of GPT4all technical report?', qa_chain)
p.log("First answer ")
answer('How was the GPT4All-J model trained?', qa_chain)
p.log("Second answer ")
answer('What was the cost of training the GPT4all model?', qa_chain)
p.log("Third answer ")

