# examples/graph_based_planner.py

import sys
import os

# Add the parent directory to sys.path to allow imports from the framework
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from agent_core.agents import Agent
from agent_core.planners import GraphPlanner


def main():
    agent = Agent()
    agent.planner = GraphPlanner()
    agent.enable_evaluators()

    task = "3 steps draw a digital dragon using computer emoji characters."
    execution_result = agent.execute(task)
    agent.get_reasoning()
    print(f"Execution Result: {execution_result}")


if __name__ == "__main__":
    main()
