from langchain.memory.chat_message_histories import MongoDBChatMessageHistory
import os, uuid
import pymongo
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from agents import generate_agent007, generate_agent008

load_dotenv()
# Set up the message history
def set_mongodb(session_id):
    connection_string = os.getenv("MONGO_CONNECTION_STRING")
    message_history = MongoDBChatMessageHistory(
        connection_string=connection_string, session_id=session_id
    )
    return message_history

if __name__ == "__main__":
    load_dotenv()
    session_id = str(uuid.uuid4())
    message_history = set_mongodb(session_id)
    myclient = pymongo.MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
    mydb = myclient["chat_history"]
    mycol = mydb["message_store"]
    while True:
        with get_openai_callback() as cb:
            user_input = input("Du: ")
            if user_input == "exit":
                break
            message_history.add_user_message(user_input)
            try: 
                agent = generate_agent007(message_history)
                answer = agent.run(user_input)
                print(answer)
                message_history.add_ai_message(answer)
            except Exception as err:
                print('Exception occured.', str(err))
        dict = vars(cb)
        dict['session_id'] = session_id
        mycol.insert_one(dict)
        print(cb)