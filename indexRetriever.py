# functions to retrieve the documents and answer the questions
import logging, sys, os
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_community.embeddings import OpenAIEmbeddings
# from llama_index.vector_stores.pinecone import PineconeVectorStore
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

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
    load_dotenv()
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
    
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    
    # initialize embedding model
    embed = OpenAIEmbeddings(
        model = "text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    text_field = "text"
    # connect to index
    vector_store = PineconeVectorStore(index, embed)

    answer = vector_store.similarity_search(question, k=5)
    return answer

def question_answer(question, doc):
    llm = ChatOpenAI(temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    chain = load_qa_chain(llm, chain_type="stuff")
    
    return chain.run(input_documents=doc, question=question)

def summarization(text):
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(text)
    
    docs = [Document(page_content=t) for t in texts]
    
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return summary

def doc_retriever(prompt):
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
    
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    
    # initialize embedding model
    embed = OpenAIEmbeddings(
        model = "text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # connect to index  
    vector_store = PineconeVectorStore(index, embed)
    
    retriever = vector_store.as_retriever()
    docs = retriever.invoke(prompt)
    print(docs)
    
    return docs

def answer_retriever(prompt):

    SYSTEM_TEMPLATE = """As a tutor for the lecture databases and informationssystems, your goal is to provide accurate and helpful infomration about the lecture. 
    You should answer the user inquiries as best as possible based on the context and chat history provided and avoid making up answers. 
    If you don't know the answer, simply state that you don'k know. Answer the question in german language. 
     
    {context}

    Question: {prompt}
    """
    # TUTOR_PROMPT = PromptTemplate(
    #     template=prompt_template, input_variables=["context", "question"]
    # )
    
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
    
    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, question_answering_prompt)
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm, 
    #     chain_type="stuff", 
    #     retriever=retriever, 
    #     chain_type_kwargs={"prompt": TUTOR_PROMPT},
    # )

    return document_chain

def question_generator(prompt):
    
    prompt_template = """Als Tutor für die Datenbanken und Informationssysteme hilfst du den Studierenden bei Übungsaufgaben. 
    Der Student wird die nach einer Übungsaufgabe zu einem speziellen Thema fragen.
    Du generierst eine Frage, die sich auf das Thema bezieht. Die Frage sollte in deutscher Sprache sein.
    
    {context}

    Input: {prompt}
    """
    # TUTOR_PROMPT = PromptTemplate(
    #     template=prompt_template, input_variables=["context", "question"]
    # )
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
    
    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                prompt_template,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm, 
    #     chain_type="stuff", 
    #     retriever=retriever, 
    #     chain_type_kwargs={"prompt": TUTOR_PROMPT},
    # )
    document_chain = create_stuff_documents_chain(llm, question_answering_prompt)
    return document_chain