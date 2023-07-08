from flask import Flask, request, render_template
from langchain.memory.chat_message_histories import MongoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory
import os, uuid, pymongo, sys, subprocess
from langchain.callbacks import get_openai_callback
from agents import generate_agent007, generate_agent009
from questionGenerator import question_generator
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()
myclient = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
mydb = myclient["chat_history"]
mycol = mydb["message_store"]

@app.route("/")
def main():
    return "nothing here"

def set_mongodb(session_id):
    connection_string = os.getenv("MONGO_CONNECTION_STRING")
    message_history = MongoDBChatMessageHistory(
        connection_string=connection_string, session_id=session_id
    )
    return message_history

@app.route("/generateQuestions")
def generateQuestions():
    client = MongoClient(os.getenv('MONGO_CONNECTION_STRING'))
    database_name = "DBISquestions"
    database_list = client.list_database_names()
    if database_name in database_list:
        client.close()
    else: 
        question_generator()
        client.close()      
    return "Questions generated"

@app.route("/chat", methods=['POST'])
def chat():
    user_input = request.json.get('msg')
    session_id = str(uuid.uuid4())
    message_history = set_mongodb(session_id)
        
    with get_openai_callback() as cb:
        message_history.add_user_message(user_input)
        try: 
            agent = generate_agent007(message_history)
            answer = agent.run(user_input)
            message_history.add_ai_message(answer)
            dict_cb = vars(cb)
            dict_cb['Session_id'] = session_id
            mycol.insert_one(dict_cb)
            return answer
        except Exception as err:
            return 'Exception occurred: ' + str(err)
