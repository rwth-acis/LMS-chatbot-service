from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain, SequentialChain
import sys
from dotenv import load_dotenv

def fact_check(question, answer):
    llm = OpenAI(temperature=0.3)

    dict = {"question": question, "answer": answer}
    template = """Here is the question: {question} and the answer: {answer} to the question. Make a bullet point list of the assumptions in the answer.\n\n"""
    prompt_template = PromptTemplate(input_variables=["question", "answer"], template=template)
    assumptions_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="assertions")

    template = """Here is a bullet point list of assertions:
    {assertions}
    For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
    prompt_template = PromptTemplate(input_variables=["assertions"], template=template)
    fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="facts")

    template = """In light of the above facts, how would you answer the question and explain your answer.""".format(question)
    template = """{facts}\n""" + template
    prompt_template = PromptTemplate(input_variables=["facts"], template=template)
    answer_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="output")

    overall_chain = SequentialChain(chains=[assumptions_chain, fact_checker_chain, answer_chain],
                                    input_variables=["question","answer"],
                                    output_variables=["assertions", "facts", "output"],
                                    verbose=True)

    return overall_chain(dict)

if __name__=="__main__":
    load_dotenv()
    while True:
        print()
        question = input("Question: ")
        answer = input("Answer: ")
        response = fact_check(question,answer)
        print(response)
    
    