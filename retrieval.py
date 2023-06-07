import logging, sys, os
import weaviate
import json
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, GPTVectorStoreIndex
from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain, ChatVectorDBChain
from dotenv import load_dotenv

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# initialize vector database
load_dotenv()

client = weaviate.Client(
    url="https://lms-service-vcwfqxu9.weaviate.network",
    auth_client_secret=weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")),
    additional_headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    }
)

class_name = "Slides"

vectorstore = Weaviate(client, "slides", 'text')

MyOpenAI = OpenAI(temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY"))

qa = ChatVectorDBChain.from_llm(MyOpenAI, vectorstore)
chat_history=[]

while True:
    query = input("")
    result = qa({"question":query, "chat_history":chat_history})
    print(result["answer"])
    chat_history.append([query, result["answer"]])

    
