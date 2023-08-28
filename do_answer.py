# Configure if CUDA should be enabled, see TROUBLESHOOTING for details on installing CUDA enabled libraries
CUDA = False

# Configure main model
#MODEL='lmsys/fastchat-t5-3b-v1.0'
MODEL='TheBloke/stable-vicuna-13B-HF'

# Vector DB directory to read embeddings from
# Model used to create the vector db embeddings
VECTOR_DIR = 'db'
EMBEDDINGS_MODEL = "hkunlp/instructor-xl"

# Import configuration for tokenizers
TOKENIZER_MODULE = 'transformers'
TOKENIZER_CLASS = 'AutoTokenizer'

# Import configuration for model
MODEL_MODULE = 'transformers'
MODEL_CLASS = 'AutoModelForSeq2SeqLM'

import profile
p = profile.Profile()
import class_loader
loader = class_loader.Loader()
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

from langchain.embeddings import HuggingFaceInstructEmbeddings
if CUDA :
	instructor_embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDINGS_MODEL, model_kwargs={"device": "cuda"})
	p.log("Initializing embeddings with CUDA ")
else :
	instructor_embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDINGS_MODEL)
	p.log("Initializing embeddings without CUDA ")
embedding = instructor_embeddings
persist_directory = VECTOR_DIR
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



#TODO nao basta mudar a string com o nome do modelo, mas a classe tambem
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0,
}

from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
tokenizer = LlamaTokenizer.from_pretrained("TheBloke/stable-vicuna-13B-HF")
p.log("Tokenizer ")
model = LlamaForCausalLM.from_pretrained(
	"TheBloke/stable-vicuna-13B-HF",
	load_in_8bit=True,
	torch_dtype=torch.float32,
	device_map='auto',
	offload_folder='offload',
	quantization_config=quantization_config,
)
p.log("Model loaded ")

#tokenizer_class = loader.load(TOKENIZER_MODULE, TOKENIZER_CLASS)
#tokenizer = tokenizer_class.from_pretrained(MODEL)
#p.log("Tokenizer ")

#model_class = loader.load(MODEL_MODULE, MODEL_CLASS)
#model = model_class.from_pretrained(MODEL)
#p.log("Model loaded ")

#pipe = pipeline(
#    "text2text-generation",
#    model=model, 
#    tokenizer=tokenizer, 
#    max_length=256
#)

pipe = pipeline(
    "text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=2048,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15
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
    print(f"\n---\n{query}")
    process_llm_response(qa_chain(query))

print(f'Using {MODEL} as model, {EMBEDDINGS_MODEL} for embeddings and {VECTOR_DIR} as Vector DB DIR')
    
answer('who are the authors of GPT4all technical report?', qa_chain)
p.log("First answer ")
answer('How was the GPT4All-J model trained?', qa_chain)
p.log("Second answer ")
answer('What was the cost of training the GPT4all model?', qa_chain)
p.log("Third answer ")

