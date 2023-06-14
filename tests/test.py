import os

import logging
import sys
from langchain.chat_models import ChatOpenAI
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, GPTVectorStoreIndex 
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# define LLM
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)

# load data and build index
DBIS_slides = SimpleDirectoryReader('../src/sorted_documents').load_data()
print(DBIS_slides[0].text)
vector_index = GPTVectorStoreIndex.from_documents(DBIS_slides, service_context=service_context)

# build query engine from vector index
query_engine_vector = vector_index.as_query_engine(similarity_top_k=3)
response = query_engine_vector.query("What is transactionmanagement?")
print(response)
