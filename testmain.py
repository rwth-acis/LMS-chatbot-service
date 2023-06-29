from langchain.memory import MongoDBChatMessageHistory
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain.tools import StructuredTool
from langchain.callbacks import get_openai_callback
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from prompts import split_prompt_question, greeting
from indexRetriever import doc_retrieval, question_retrieval, question_answer, summarization, test
from factchecker import fact_check
import re, os, logging, sys
from dotenv import load_dotenv

def answerQuestion(question):
    question_split = split_prompt_question(question)
    subquestion = question_retrieval(question_split)
    subquestion_answer = ""
    for i in range(len(subquestion)):
        doc = doc_retrieval(subquestion[i])
        answers = question_answer(subquestion[i], doc)
        subquestion_answer += answers
    # summarize answers
    overall_answer = summarization(subquestion_answer)
    response = fact_check(question, overall_answer)
    return response




if __name__=="__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    
    load_dotenv()
    tools = [
        Tool(
            name="answerQuestion",
            func=test,
            description="call this to answer the user's question.",
        ),
        Tool(
            name="greet",
            func=greeting,
            description="call this to greet the user.",  
        ),
        
    ]

    # Set up the base template
    template = """Complete the objective as best you can. You have access to the following tools:

    {tools}

    Use the following format for the input:

    Greet: greet the user using the greeting tool
    Input: the input you must answer. 
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat 3 times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    These were previous tasks you completed:



    Begin!

    Input: {input}
    {agent_scratchpad}"""

    # Set up a prompt template
    class CustomPromptTemplate(BaseChatPromptTemplate):
        # The template to use
        template: str
        # The list of tools available
        tools: List[Tool]
        
        def format_messages(self, **kwargs) -> str:
            # Get the intermediate steps (AgentAction, Observation tuples)
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            formatted = self.template.format(**kwargs)
            return [HumanMessage(content=formatted)]
        
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )

    class CustomOutputParser(AgentOutputParser):
        
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            # Check if agent should finish
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            # Parse out the action and action input
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
        
    output_parser = CustomOutputParser()
        
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    llm_chain = LLMChain(llm=llm, prompt = prompt)
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )
    
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    connection_string = "mongodb://localhost:27017"

    chat_history = MongoDBChatMessageHistory(
        connection_string=connection_string, session_id="test_session"
    )

    while True:
        print()
        with get_openai_callback() as cb:
            input = input("you: ")
            if input == "exit":
                break
            chat_history.add_user_message(input)
            answer = agent_executor.run(input)
            print(answer)
            chat_history.add_ai_message(answer)
        print(cb)