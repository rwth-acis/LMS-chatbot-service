import logging, sys, os
import pinecone
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, GPTVectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone, Chroma
from dotenv import load_dotenv

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

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
        dimension=1536
    )

# connect to index
index = pinecone.GRPCIndex(index_name)

slides = SimpleDirectoryReader('src/documents').load_data()
docsearch = Pinecone.from_texts([t.text for t in slides], embed, index_name=index_name)

#query = "What is DBIS?"
#docs = docsearch.similarity_search(query, k=1)
#print(docs)
llm = ChatOpenAI(temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY"))
chain = load_qa_chain(llm, chain_type="stuff")

query = "What is transactionmanagement?"
docs = docsearch.similarity_search(query, k=1)
answer = chain.run(input_documents=docs, question=query)
print(answer)