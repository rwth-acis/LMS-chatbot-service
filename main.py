from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain.agents.tools import Tool
from langchain import LLMMathChain, LLMChain
from langchain.agents import ZeroShotAgent, AgentExecutor, BaseMultiActionAgent, BaseSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_memory import ChatMessageHistory
from langchain.memory.chat_message_histories import MongoDBChatMessageHistory
from indexRetriever import answer_retriever
from prompts import greeting
import os, uuid, json, ast
from typing import List, Tuple, Any, Union
import pymongo
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback

load_dotenv()
# Set up the message history
def set_mongodb(session_id):
    connection_string = os.getenv("MONGO_CONNECTION_STRING")
    message_history = MongoDBChatMessageHistory(
        connection_string=connection_string, session_id=session_id
    )
    return message_history

# def generate_agent(message_history):
#     llm = OpenAI(temperature=0)
#     tools = [
#         Tool(
#             name="answerQuestion",
#             func=answer_retriever,
#             description="call this to answer the user's question.",
#         ),
#     ]

#     prefix = """You are a tutor for the lecture databases and informationssystems. Have a conversation with a student of this lecture, answering the following questions as best you can. Only use the following tools:"""
#     suffix = """Begin! Remember to answer in german.

#     {chat_history}
#     Question: {input}
#     {agent_scratchpad}"""

#     prompt = ZeroShotAgent.create_prompt(
#         tools,
#         prefix=prefix,
#         suffix=suffix,
#         input_variables=["input", "chat_history", "agent_scratchpad"],
#     )

#     memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history)
#     llm_chain = LLMChain(llm=llm, prompt=prompt)
#     agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
#     agent_chain = AgentExecutor.from_agent_and_tools(
#         agent=agent, tools=tools,verbose=True, memory=memory
#     )
#     return agent_chain
    
class QuestionAgent(BaseSingleActionAgent):
    """Fake Custom Agent."""

    @property
    def input_keys(self):
        return ["input"]

    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        return AgentAction(tool="answerQuestion", tool_input=kwargs["input"], log="")

    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        return AgentAction(tool="answerQuestion", tool_input=kwargs["input"], log="")

def generate_agent007(message_history):
    tools = [
        Tool(
            name="answerQuestion",
            func=answer_retriever,
            description="call this to answer the user's question.",
            return_direct=True,
        ),
    ]

    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history)

    agent = QuestionAgent()
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
    return agent_executor

if __name__ == "__main__":
    load_dotenv()
    session_id = str(uuid.uuid4())
    message_history = set_mongodb(session_id)
    myclient = pymongo.MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
    mydb = myclient["chat_history"]
    mycol = mydb["message_store"]
    while True:
        with get_openai_callback() as cb:
            user_input = input("Du: ")
            if user_input == "exit":
                break
            message_history.add_user_message(user_input)
            try: 
                agent = generate_agent007(message_history)
                answer = agent.run(user_input)
                print(answer)
                message_history.add_ai_message(answer)
            except Exception as err:
                print('Exception occured.', str(err))
        dict = vars(cb)
        dict['session_id'] = session_id
        mycol.insert_one(dict)
        print(cb)