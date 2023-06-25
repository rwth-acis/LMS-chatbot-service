from langchain.schema import HumanMessage, AIMessage
from prompts import split_prompt_question
from pineconeDatabase import answer_retrieval, question_retrieval
from factchecker import fact_check
from dotenv import load_dotenv

if __name__=="__main__":
    load_dotenv()
    
    chat_history = []
    
    while True:
        print()
        questions = input("Question: ")
        question_split = split_prompt_question(questions)
        subquestion = question_retrieval(question_split)
        for i in range(len(subquestion)):
            answer = answer_retrieval(subquestion[i])
            chat_history.append(HumanMessage(content=subquestion[i]))
            chat_history.append(AIMessage(content=answer))
        print(chat_history)
        