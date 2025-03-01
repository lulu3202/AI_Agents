from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessageGraph
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
import os
from dotenv import load_dotenv


load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatOpenAI(temperature=0)


def make_default_graph():
    """Make a simple LLM agent"""
    graph_workflow = StateGraph(State)
    def call_model(state):
        return {"messages": [model.invoke(state["messages"])]}

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.add_edge(START, "agent")

    agent = graph_workflow.compile()
    return agent

def make_alternative_graph():
    """Make a tool-calling agent"""

#define tool
    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

#add to tool node
    tool_node = ToolNode([add])

#bind it with the LLM 
    model_with_tools = model.bind_tools([add])
    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

#creating a function to check if it should go to tools or not. Tool node works in this function
    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)

    agent = graph_workflow.compile()
    return agent

#calling this fucntion which means it is going to execute this function 
agent=make_alternative_graph()

#To track this in LangSmith, a json file is needed. Inside this dependencies are provided per LangSmith documentation. 
#.dot  means current directory 
# The last line agent is also referenced there ../env means go to parent directory
#for langgraph studio, you need to import langgraph-cli library
#To run this, go to cd 6-Debugging folder and type "langgraph dev" to execute this directly in Langsmith - automatically langsmith opens up 