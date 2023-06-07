import logging, sys, os
import weaviate
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, GPTVectorStoreIndex
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.storage.storage_context import StorageContext
from langchain import OpenAI
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

class_obj = {
  "class": "slides",
  "vectorizer": "text2vec-openai",
  "moduleConfig": {
    "text2vec-openai": {
      "model": "ada",
      "modelVersion": "002",
      "type": "text"
    }
  }
}

slides = SimpleDirectoryReader('src/documents').load_data()
vector_store = WeaviateVectorStore(weaviate_client=client, class_prefix="Slides")
storage_context=StorageContext.from_defaults(vector_store=vector_store)
slides_index = GPTVectorStoreIndex.from_documents(slides, storage_context=storage_context)

client.schema.get(class_obj)