import logging, sys, os
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def question_retrieval(input):
    subquestion = []
    for i in range(len(input)):
        index = input[i].find('?')
        if index == -1:
            i+1
        else:
            subquestion.append(input[i])
    return subquestion
    
def doc_retrieval(question):
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

    answer = vector_store.similarity_search(question, k=5)
    return answer

def question_answer(question, doc):
    llm = ChatOpenAI(temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    chain = load_qa_chain(llm, chain_type="stuff")
    
    return chain.run(input_documents=doc, question=question)
