# examples/langgraph_integration.py

import sys
import os

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from agent_core.agents.langgraph_agent import AgentState, agent_execute
from agent_core.planners import GraphPlanner

# Add the parent directory to sys.path to allow imports from the framework
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


if __name__ == "__main__":
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent_core", agent_execute)
    graph_builder.add_edge(START, "agent_core")
    graph_builder.add_edge("agent_core", END)
    graph = graph_builder.compile()
    print("Graph build success")

    task = "draw a flower"

    print("Agent Basic")
    agent_config_basic = {
        "messages": [{"role": "user", "content": task}]
    }
    for event in graph.stream(agent_config_basic):
        for value in event.values():
            print("Assistant:", value["messages"])

    print("Agent Advanced")
    agent_config_advanced = {
        "messages": [{"role": "user", "content": task}],
        "planner": GraphPlanner(),
        "evaluators_enabled": True,
        "model_name": "gemini-1.5-flash-002",
        "knowledge": "To draw a object you need to take 3 steps, 1) prepare tools, 2) prepare "
                     "paper, 3) draw the object",
        "background": "You are a professional artist"
    }
    for event in graph.stream(agent_config_advanced):
        for value in event.values():
            print("Assistant:", value["messages"])
