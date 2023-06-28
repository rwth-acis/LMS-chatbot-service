from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain, SequentialChain

def greeting(input):
    llm = OpenAI(temperature=0)
    prompt_template = "You are a tutor for the lecture databases and information systems at the RWTH Aachen University. The students greets you: {input}. Greet the student back as a tutor would do and start some small talk."
    prompt=PromptTemplate(input_variables=["input"], template=prompt_template)
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
    )
    return llm_chain.run(input)


def split_prompt_question(input):
    llm = OpenAI(temperature=0)
    
    template = """You will get the following input: \n\n{input}\n\n. If you get a question and an assumption, split them up into Questions and Assumptions."""
    prompt_template = PromptTemplate(input_variables=["input"], template=template)
    question_chain = LLMChain(llm=llm, prompt=prompt_template)
    
    template = """Here is a question and an assumption from the student:\n\n{question}\n\n.
    Split the questions and assumptions up into subquestions and subassumption. Do not answer the questions or correct the assumptions.
    The answer should look like this:
    
    Questions:
    1. first subquestion
    2. second subquestion
    3. ...
    Assumptions:
    1. first subassumption
    2. secon subassumption
    3. ...
    """
    prompt_template = PromptTemplate(input_variables=["question"], template=template)
    subquestions_chain = LLMChain(llm=llm, prompt=prompt_template)
    
    question_chain = SimpleSequentialChain(chains=[question_chain, subquestions_chain],
                                           verbose=True)
    questions = question_chain.run(input)
    
    return questions.splitlines(True)
