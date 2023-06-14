from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
import sys
from dotenv import load_dotenv

def fact_check(questionAnswer):
    llm = OpenAI(temperature=0.3)

    template = """Here is the question and the answer to the question. Make a bullet point list of the assumptions in the answer.\n\n"""
    prompt_template = PromptTemplate(input_variables=["questionAnswer"], template="f{questionAnswer}\n\n template")
    assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)

    template = """Here is a bullet point list of assertions:
    {assertions}
    For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
    prompt_template = PromptTemplate(input_variables=["assertions"], template=template)
    fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)

    template = """In light of the above facts, how would you answer the question.""".format(questionAnswer)
    template = """{facts}\n""" + template
    prompt_template = PromptTemplate(input_variables=["facts"], template=template)
    answer_chain = LLMChain(llm=llm, prompt=prompt_template)

    overall_chain = SimpleSequentialChain(chains=[assumptions_chain, fact_checker_chain, answer_chain], verbose=True)

    return overall_chain.run(questionAnswer)

if __name__=="__main__":
    load_dotenv()
    while True:
        print()
        questionAnswer = input("Question and Answer: ")
        answer = fact_check(questionAnswer)
        print(answer)
    
    