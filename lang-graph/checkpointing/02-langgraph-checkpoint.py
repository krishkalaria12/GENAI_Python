import os

from typing_extensions import TypedDict
from typing import Annotated

from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver

api_key = os.getenv("GEMINI_API_KEY")


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.0-flash",
    api_key=api_key,
)


def chat_node(state: State):
    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }


graph_builder = StateGraph(State)

graph_builder.add_node("chat_node", chat_node)
graph_builder.add_edge(START, "chat_node")
graph_builder.add_edge("chat_node", END)


def compile_graph_with_checkpointer(checkpointer):
    graph_with_checkpointer = graph_builder.compile(checkpointer=checkpointer)
    return graph_with_checkpointer


def main():
    # MongoDB connection details
    DB_URI = "mongodb://admin:admin@localhost:27017"

    # Config for thread_id
    config = {
        "configurable": {
            "thread_id": 1,
        }
    }

    with MongoDBSaver.from_conn_string(DB_URI) as mongo_checkpointer:
        graph_with_mongo = compile_graph_with_checkpointer(mongo_checkpointer)

        query = input("> ")

        _state = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }

        result = graph_with_mongo.invoke(_state, config)

        print("result:", result)


main()
