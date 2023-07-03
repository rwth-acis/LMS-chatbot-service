# this script generates questions out of the inserted documents. (currenty 102 question-answers)
# It is not using the pinecone index since there is no possibility to retrieve all documents from the pinecone database.
# The questions and answers are stored in a MongoDB database.
# this will be especially used to question the students as a chatbot and test their knowledge.

from llama_index import SimpleDirectoryReader
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain
from pymongo import MongoClient
import requests
from dotenv import load_dotenv
import os
from langchain.callbacks import get_openai_callback

def question_generator():
    load_dotenv()
    # connection to the mongodb database
    client = MongoClient(os.getenv('MONGO_CONNECTION_STRING'))

    # select database
    db = client["DBISquestions"]

    # select collection
    collection = db["questionAnswer"]
    
    # get all documents from the src/documents folder
    DBIS_slides = SimpleDirectoryReader('src/documents').load_data()
    
    # retrieve texts from the documents
    text = ""
    for i in range(len(DBIS_slides)):
        text += DBIS_slides[i].text
    
    # generate questions out of the text
    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
    qa = chain.run(text)
    
    # insert questions and answers into the database
    for i in range(len(qa)):
        collection.insert_one(qa[i])
    return 
