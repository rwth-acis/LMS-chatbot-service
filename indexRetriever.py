import logging, sys, os
import pinecone, openai
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, GPTVectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import PineconeVectorStore
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
    pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
    )
    
    index = pinecone.Index('dbis-slides')
    
    # initialize embedding model
    embed = OpenAIEmbeddings(
        model = "text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    text_field = "text"
    # connect to index
    vector_store = Pinecone(index, embed.embed_query, text_field)
    # docsearch = Pinecone.from_texts([t.text for t in slides], embed, index_name=index)

    # llm = ChatOpenAI(temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY"))

    # retriever_func = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

    # tool = [
    #     Tool(
    #         name="dbis_slides",
    #         func=retriever_func.run,
    #         description="useful for when you need to answer a question about the lecture Databases and Information Systems. Input should be a question about the lecture.")
    # ]

    # agent = initialize_agent(tool, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    answer = vector_store.similarity_search(question, k=5)
    return answer

if __name__ =="__main__":
    load_dotenv()
    while True:
        print()
        question = input("Input: ")
        response = answer_retrieval(question)
        print(response)
        