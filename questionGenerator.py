# this script generates questions out of the inserted documents. (currenty 102 question-answers)
# It is not using the pinecone index since there is no possibility to retrieve all documents from the pinecone database.
# The questions and answers are stored in a MongoDB database.
# this will be especially used to question the students as a chatbot and test their knowledge.

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from llama_index import SimpleDirectoryReader
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain
from pymongo import MongoClient
import requests
from dotenv import load_dotenv
import os, random
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
    slides = SimpleDirectoryReader(os.getenv("SELECTED_FILES")).load_data()
    
    # retrieve texts from the documents
    text = ""
    for i in range(len(slides)):
        text += slides[i].text
    
    # generate questions out of the text
    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
    qa = chain.run(text)
    
    # insert questions and answers into the database
    for i in range(len(qa)):
        collection.insert_one(qa[i])
    return 

def random_question_tool(input):
    client = MongoClient(os.getenv('MONGO_CONNECTION_STRING'))
    db = client["DBISquestions"]
    col = db["questionAnswer"]
    questionanswers = col.find()
    a = random.randint(0, 102)
    question = questionanswers[a].get("question")
    
    llm = OpenAI(temperature=0)
    template = """Here is a question: {question}. Translate the question into german"""
    prompt_template = PromptTemplate(input_variables=["question"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt_template, output_key="german_question")
    return chain.run(question)
