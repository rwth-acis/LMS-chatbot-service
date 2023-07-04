# first attempts of custom agents

from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain.agents.tools import Tool
from langchain import LLMMathChain, LLMChain, PromptTemplate
from langchain.agents import AgentType, AgentExecutor, ZeroShotAgent, ConversationalChatAgent, BaseMultiActionAgent, BaseSingleActionAgent, LLMSingleActionAgent, AgentOutputParser, initialize_agent
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory, ConversationBufferWindowMemory
from langchain.memory.chat_memory import ChatMessageHistory
from typing import List, Tuple, Any, Union
from indexRetriever import answer_retriever
from questionGenerator import random_question_tool
from factchecker import fact_check
import re
from dotenv import load_dotenv

class QuestionAgent(BaseSingleActionAgent):

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
        Tool(
            name="tutorQuestion",
            func=random_question_tool,
            description="call this to generate a question for the user to answer.",
            return_direct=True,
        )
    ]

    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history)

    agent = QuestionAgent()
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
    return agent_executor

def generate_agent009():
    tools = [
        Tool(
            name="answerQuestion",
            func=answer_retriever,
            description="call this to answer the user's question.",
            return_direct=True,
        ),
        Tool(
            name="tutorQuestion",
            func=random_question_tool,
            description="call this to generate a question for the user to answer if the user wants to study for the lecture.",
            return_direct=True,
        )
    ]

    llm = ChatOpenAI(temperature=0)
    # memory = ConversationBufferWindowMemory(k=10)
    agent_chain = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
        verbose=True,
        # memory=memory,
    )
    
    return agent_chain

