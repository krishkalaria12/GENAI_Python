from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.schema import SystemMessage

# custom imports 
from .config import llm_with_tools, tools, SYSTEM_PROMPT


class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    sys_prompt = SystemMessage(content=SYSTEM_PROMPT)
    message = llm_with_tools.invoke([sys_prompt] + state["messages"])
    assert len(message.tool_calls) <= 1

    return { "messages": [message] }

tool_node = ToolNode(tools=tools)
graph_builder = StateGraph(State)

# add nodes to the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# add edges to the graph
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

def compile_graph_with_checkpointer(checkpointer):
    graph_with_checkpointer = graph_builder.compile(checkpointer=checkpointer)
    return graph_with_checkpointer