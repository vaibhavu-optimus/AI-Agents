# ReAct = Reasoning + Acting

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage # Foundational message class for all message types in langgraph
from langchain_core.messages import ToolMessage # Passes data back to the LLM after it calls a tool such as the content and tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import Tool
from langgraph.graph.message import add_messages # Reducer function
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os

load_dotenv()

# Annotated - provides additional context without affecting the type itself
# email: Annotated[str, "The email address of the user"]
# print(email.__metadata__)  # Output: {'description': 'The email address of the user'}

# Sequence - To automatically handle the state updates for sequences such as by adding new messages
# to a chat history

# add_messages - A reducer function that takes the current state and a new message, and returns the updated state with the new message added to the messages list. Rule that controls how updates from nodes are combined with the existing state. 
# Tells us how to merge new data into the current state
# Without a reducer, updates would have rather replaced the existing value entirely.

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] # The state of the agent, which is a list of messages. The add_messages reducer function is used to automatically handle the state updates for sequences such as by adding new messages to a chat history.

@tool
def add(a: int, b: int) -> int:
    """This function returns sum of two numbers"""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """This function returns difference of two numbers"""
    if(a < b):
        swap(a, b)
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """This function returns product of two numbers"""
    return a * b

tools = [add, subtract, multiply]

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-01-preview"),
    temperature=0,
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )

    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": response}

def should_continue(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls :
        return "end"
    else :
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edge(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "exit": END
    }
)

graph.add_edge("tools", "agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if(isinstance(message, tuple)):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 34 + 21, Add 3 + 4")]}
print_stream(app.stream(inputs, stream_mode="values"))