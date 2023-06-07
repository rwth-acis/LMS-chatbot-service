import os

import logging
import sys

from langchain.chat_models import ChatOpenAI
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, GPTVectorStoreIndex 
from llama_index.response.pprint_utils import pprint_response
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from langchain import OpenAI
from IPython.display import Markdown, display
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# define LLM
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# load data and build index
DBIS_slides = SimpleDirectoryReader('src/sorted_documents').load_data()
# vector_index = GPTVectorStoreIndex.from_documents(DBIS_slides)

graph_index = GPTKnowledgeGraphIndex.from_documents(
    DBIS_slides, 
    max_triplets_per_chunk=2,
    service_context=service_context
)

# build query engine from vector index
# query_engine_vector = vector_index.as_query_engine(similarity_top_k=3)
# response = query_engine_vector.query("What is DBIS?", response_mode="summary")
# print(response)

# build query engine from graph index
query_engine_graph = graph_index.as_query_engine(similarity_top_k=3)
response = query_engine_graph.query("What is DBIS?")
print(response)