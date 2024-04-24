# used to initialize the pinecone index and upload the documents to the pinecone index
# the index is used to retrieve the documents and answer the questions

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, StorageContext, set_global_service_context
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from pathlib import Path
import tiktoken
from pinecone import Pinecone, ServerlessSpec
import openai
import logging
import sys
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


llm = OpenAI(model="gpt-4", temperature=0, max_tokens=256)

embed = OpenAIEmbedding(
    model="text-embedding-ada-002", 
    openai_api_key=os.getenv("OPENAI_API_KEY"))

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-4").encode
)

callback_manager = CallbackManager([token_counter])

set_global_service_context(
    ServiceContext.from_defaults(
        llm=llm, 
        embed_model=embed, 
        callback_manager=callback_manager
    )
)

index_name = os.getenv("PINECONE_INDEX_NAME")

# pinecone.init(
#     api_key=os.getenv("PINECONE_API_KEY"),
#     environment=os.getenv("PINECONE_ENVIRONMENT")
# )
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

#index = pinecone.GRPCIndex(index_name)

parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed, 
    chunk_size=1024,
    callback_manager=callback_manager
)

storage_context = StorageContext.from_defaults(
    vector_store=PineconeVectorStore(pc.Index(index_name)),
)

slides = SimpleDirectoryReader(os.getenv("SELECTED_FILES")).load_data()
exercises = SimpleDirectoryReader(os.getenv("SELECTED_EXERCISES")).load_data()

nodes_slides = parser.get_nodes_from_documents(slides)
nodes_exercises = parser.get_nodes_from_documents(exercises)
storage_context.docstore.add_documents(nodes_slides)
storage_context.docstore.add_documents(nodes_exercises)

slides_index = VectorStoreIndex.from_documents(
    slides, 
    service_context=service_context, 
    storage_context=storage_context)

exercise_index = VectorStoreIndex.from_documents(
    exercises,
    service_context=service_context,
    storage_context=storage_context)

print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
)
