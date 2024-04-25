from langchain_openai import ChatOpenAI
from langchain.agents.tools import Tool
from langchain_core.messages import HumanMessage
# from langchain.chains.LLMMathChain import LLMMathChain, LLMChain, PromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate
from indexRetriever import answer_retriever, question_generator
from weaviateRetriever import get_docs
from questionGenerator import random_question_tool
from factchecker import fact_check

@tool
def get_answer(prompt):
    """Rufe diese Funktion auf, um alle Fragen zur Vorlesung Datenbanken und Informationssysteme zu beantworten."""
    docs = get_docs(prompt)
    answer_retrieve = answer_retriever()
    answer = answer_retrieve.invoke({        
                            "context": docs,
                            "prompt": prompt,
                            "messages": [
                                HumanMessage(content=prompt)
                            ],
                        })
    return str(answer)

@tool
def get_question(prompt):
    """Rufe die Funktion auf, um eine Frage zu einem spezifischen Thema zu generieren. Übersetze sie aber davor ins Deutsche."""
    docs = get_docs(prompt)
    question_retriever = question_generator()
    question = question_retriever.invoke({        
                            "context": docs,
                            "prompt": prompt,
                            "messages": [
                                HumanMessage(content=prompt)
                            ],
                        })
    
    return str(question)

@tool
def random_question(prompt):
    """Dieses Tool gibt eine zufällige Frage wieder, die sich auf die Vorlesung Datenbank und Informationssysteme bezieht. Beantworte die Frage, die vom Tool wiedergegeben wurde nicht, sondern stelle dem Studenten diese Frage."""
    question = random_question_tool(prompt)
    q = f"Die Frage lautet: {question}"
    return q

def generate_agent007(memory):
    
    tools = [
        random_question,
        get_answer,
        get_question
    ]
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Benutze immer ein tool für die Beantwortung der Fragen. Du bist Tutor für die Vorlesung Datenbanken und Informationssysteme an der Universität RWTH Aachen. Dein Ziel ist die Studenten bei der Vorlesung zu unterstützen und Fragen zur Vorlesung zu beantworten, indem du eine Konversation mit ihnen führst. Ebenso kannst du den Studierenden Übungsaufgaben stellen und ihre Antworten korrigieren. Da die Vorlesung auf Deutsch gehalten wird, solltest du auch auf Deutsch antworten. Du kannst nur Fragen zur Vorlesung beantworten. Die Antwort auf Inhalte, die nicht Teil der Vorlesung sind, sollen verweigert werden. Sei immer freundlich und wenn du eine Frage nicht beantworten kannst, gib dies zu. Zusammengefasst ist der Tutor ein mächtiges System, das bei einer Vielzahl von Aufgaben helfen kann und wertvolle Einblicke und Informationen zu einer Vielzahl von Themen liefern kann. Ob du Hilfe bei einer bestimmten Frage benötigst oder einfach nur ein Gespräch über ein bestimmtes Thema führen möchtest, der Tutor ist hier, um zu helfen.",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    llm = ChatOpenAI(temperature=0, model="gpt-4")
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    # agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, system_message=prefix, human_message=suffix, tools=tools, verbose=True)
    agent_chain = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        memory=memory,
        return_intermediate_steps=True, 
        handle_parsing_errors=True,
    )

    return agent_chain

