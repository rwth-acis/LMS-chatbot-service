import sys, os
import logging
import weaviate

from pypdf import PdfReader
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, GPTVectorStoreIndex 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate

def extract_text_from_pdf(file_path, header_lines):
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        text = ''
        for i, page in enumerate(pdf.pages):
            content = page.extract_text()
            lines = content.split('\n')
            if i == 0:
                filtered_lines = lines
            else:
                filtered_lines = lines[header_lines:]
            text += '\n'.join(filtered_lines)
        return text

def extract_text_from_pdfs_in_folder(folder_path, header_lines):
    text_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(file_path, header_lines)
            text_list.append(text)
    return text_list

# text list is an array can't split into chunks
# splits text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# creates vectorstore from text chunks
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# creates conversation chain from vectorstore
def get_conversation_chain(vectorstore, model):
    llm = model
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def make_chain():
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0
    )
    folder_path = 'src/documents/'
    
    slides = extract_text_from_pdfs_in_folder(folder_path, 3)
    
    text_chunks = get_text_chunks(slides)
    
    vectorstore = get_vectorstore(text_chunks)
    
    chain = get_conversation_chain(vectorstore, model)
    
    return chain