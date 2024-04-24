from flask import Flask, request, render_template
from langchain_mongodb import MongoDBChatMessageHistory
# from langchain_community.chat_message_histories import MongoDBChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
import os
from langchain_community.callbacks import get_openai_callback
from agents import generate_agent007
from questionGenerator import question_generator
from factchecker import checker_chain
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()
myclient = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
mydb = myclient["chat_history"]
mycol = mydb["message_store"]
costcol = mydb["costs"]

def set_mongodb(session_id):
    # session_id = str(uuid.uuid4())
    connection_string = os.getenv("MONGO_CONNECTION_STRING")
    message_history = MongoDBChatMessageHistory(
        connection_string=connection_string, session_id=session_id
    )
    return message_history

@app.route("/")
def main():
    return "nothing here"

@app.route("/generateQuestions")
def generateQuestions():
    client = MongoClient(os.getenv('MONGO_CONNECTION_STRING'))
    database_name = "DBISquestions"
    database_list = client.list_database_names()
    if database_name in database_list:
        client.close()
    else: 
        with get_openai_callback() as cb:
            question_generator()
            print(cb)
            client.close()      
    return "Questions generated"

@app.route("/chat", methods=['POST'])
def chat():
    user_input = request.json.get('msg')
    session_id = request.json.get('channel')
    message_history = set_mongodb(session_id)
    #memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history, return_messages=True)
    memory = ConversationBufferWindowMemory(memory_key="chat_history", chat_memory=message_history, return_messages=True, output_key='output', k=4)
    with get_openai_callback() as cb:
        try: 
            agent = generate_agent007(memory)
            answer = agent(user_input)
            print(answer['output'])
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            print(f"Session_id: {session_id}")
            costs = dict()
            costs["Total Tokens"] = cb.total_tokens
            costs["Prompt Tokens"] = cb.prompt_tokens
            costs["Completion Tokens"] = cb.completion_tokens
            costs["Total Cost (USD)"] = cb.total_cost
            costs["session_id"] = session_id
            costcol.insert_one(costs)
            
            return answer['output']
        except Exception as err:
            return 'Exception occurred: ' + str(err)

