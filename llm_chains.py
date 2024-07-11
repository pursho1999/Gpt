from prompt_templates import memory_prompt_template#, pdf_chat_prompt
from langchain.chains import StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
#from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
#from langchain.llms import CTransformers
from langchain_community.llms import CTransformers
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
import chromadb
import yaml
import streamlit as st

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# def create_llm(model_path = config["model_path"]['large'],model_type = config["model_type"],model_config = config["model_config"]):
#     llm = CTransformers(model_path=model_path, model_type=model_type, config=model_config)
#     return llm

def create_llm(model_path = config["model_path"]['large'],model_type = config["model_type"],model_config = config["model_config"]):
    llm = CTransformers(model=model_path, model_type=model_type, config=model_config)
    return llm

# def create_embeddings(embeddings_path=config["embeddings_path"]):
#     return HuggingFaceInstructEmbeddings(model_name=embeddings_path)
def create_embeddings(embeddings_path = config["embeddings_path"]):
    return HuggingFaceInstructEmbeddings(model_name=embeddings_path)

def create_chat_memeory(chat_history):
    return ConversationBufferWindowMemory(memory_key="history",chat_memory=chat_history,k=3)

def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)
    

def create_llm_chain(llm, chat_prompt, memory):
    return LLMChain(llm=llm, prompt=chat_prompt,  memory=memory)


def load_normal_chain(chat_history):
    return chatChain(chat_history)
    
def load_vectordb(embeddings):

    persistent_client = chromadb.PersistentClient("chroma_db")


    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name="pdfs",
        embedding_function=embeddings,
    )
    return langchain_chroma

def load_pdf_chat_chain(chat_history):
    return pdfChatChain(chat_history)

def load_retrival_chain(llm, memory, vectordb):
    return RetrievalQA.from_llm(llm=llm, memory=memory, retriever=vectordb.as_retriever())

class pdfChatChain:
    def __init__(self, chat_history):
        self.memory=create_chat_memeory(chat_history)
        self.vectordb=load_vectordb(create_embeddings())
        llm=create_llm()
        #chat_prompt=create_prompt_from_template(memory_prompt_template)
        self.llm_chain=load_retrival_chain(llm, self.memory,self.vectordb)

    def run(self, user_input):
        print("PDF Chat Chain is Running...")
        return self.llm_chain.run(query = user_input,history = self.memory.chat_memory.messages , stop=["Human:"])


class chatChain:
    def __init__(self, chat_history):
        self.memory=create_chat_memeory(chat_history)
        llm=create_llm()
        chat_prompt=create_prompt_from_template(memory_prompt_template)
        self.llm_chain=create_llm_chain(llm, chat_prompt, self.memory)

    def run(self, user_input):
        return self.llm_chain.run(human_input = user_input,history = self.memory.chat_memory.messages , stop=["Human:"])