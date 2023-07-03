# used to initialize the pinecone index and upload the documents to the pinecone index
# the index is used to retrieve the documents and answer the questions

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext, StorageContext
from llama_index.vector_stores import PineconeVectorStore
import pinecone
import openai
import logging
import sys
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)

embed = OpenAIEmbeddings(
    model="text-embedding-ada-002", 
    openai_api_key=os.getenv("OPENAI_API_KEY"))

index_name = "dbis-slides"

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        metric="dotproduct",
        dimension=1536
    )

#index = pinecone.GRPCIndex(index_name)

storage_context = StorageContext.from_defaults(
    vector_store=PineconeVectorStore(pinecone.Index(index_name)),
)

#TODO add files manually
DBIS_slides = SimpleDirectoryReader('src/documents').load_data()
slides_index = GPTVectorStoreIndex.from_documents(
    DBIS_slides, 
    service_context=service_context, 
    storage_context=storage_context)
