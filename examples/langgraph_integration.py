# examples/langgraph_integration.py

import json
import sys
import os
from typing import TypedDict, Annotated

from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph

# Add the parent directory to sys.path to allow imports from the framework
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from agent_core.agents import Agent


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


def agent_execute(state: State):
    agent = Agent()
    agent.execute(state["messages"][-1].content)
    execution_result = agent.get_execution_result_summary()
    return {"messages": execution_result}


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"])


if __name__ == "__main__":
    graph_builder = StateGraph(State)
    graph_builder.add_node("agent_core", agent_execute)
    graph_builder.add_edge(START, "agent_core")
    graph_builder.add_edge("agent_core", END)
    graph = graph_builder.compile()
    stream_graph_updates("who are you")
