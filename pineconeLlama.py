import logging, sys, os
import pinecone
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, GPTVectorStoreIndex, OpenAIEmbedding
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone, Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool
import tiktoken
from dotenv import load_dotenv
from tqdm.auto import tqdm
from uuid import uuid4
import json
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

# define LLM
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))

# initialize vector database
index_name = 'dbis-slides'
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-west4-gcp"
)

# initialize embedding model
embed = OpenAIEmbeddings(
    model = "text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# create new index if not in index list
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name, 
        metric="dotproduct",
        dimension=1536, # 1536 is the dimension of the text-embedding-ada-002 model
        pod_type="p1"
    )

tokenizer = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,
        )
# connect to index
index = pinecone.GRPCIndex(index_name)

# load data and build index
slides = SimpleDirectoryReader('../src/documents').load_data()
service_context = ServiceContext.from_defaults(llm_predictor=LLMPredictor(), embed_model=OpenAIEmbedding(), chunk_size_limit=512)
storage_context = StorageContext.from_defaults(vector_store=PineconeVectorStore(pinecone.Index("slides"), tokenizer=tokenizer))
slides_index = GPTVectorStoreIndex.from_documents(slides, service_context=service_context ,storage_context=storage_context)
