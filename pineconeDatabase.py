import logging, sys, os
import pinecone
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, GPTVectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone, Chroma
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import BaseTool
from dotenv import load_dotenv

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

def question_retrieval(input):
    subquestion = []
    for i in range(len(input)):
        index = input[i].find('?')
        if index == -1:
            i+1
        else:
            subquestion.append(input[i])
    return subquestion
    
def answer_retrieval(question):
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

    llm = ChatOpenAI(temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY"))

    retriever_func = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

    tool = [
        Tool(
            name="dbis_slides",
            func=retriever_func.run,
            description="useful for when you need to answer a question about the lecture Databases and Information Systems. Input should be a question about the lecture.")
    ]

    agent = initialize_agent(tool, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return agent.run(question)

if __name__ =="__main__":
    load_dotenv()
    while True:
        print()
        input = input("Input:")
        response = question_retrieval(input)
        print(response)
        