from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from prompts import split_prompt_question
from indexRetriever import doc_retrieval, question_retrieval, question_answer, summarization
from factchecker import fact_check
from dotenv import load_dotenv

if __name__=="__main__":
    load_dotenv()
    
    chat_history = ConversationBufferMemory()
    
    while True:
        print()
        questions = input("Question: ")
        chat_history.chat_memory.add_user_message(questions)
        question_split = split_prompt_question(questions)
        subquestion = question_retrieval(question_split)
        subquestion_answer = ""
        for i in range(len(subquestion)):
            doc = doc_retrieval(subquestion[i])
            answers = question_answer(subquestion[i], doc)
            checks = fact_check(subquestion[i], answers)
            subquestion_answer += checks["answer"] 
            # summarize answers
        overall_answer = summarization(subquestion_answer)
        print(overall_answer)
        chat_history.chat_memory.add_ai_message(overall_answer)
        # history in mongodb abspeichern