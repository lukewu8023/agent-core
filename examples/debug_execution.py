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

    # Patch key methods to provide debug points for IDE breakpoints
    original_node_execute = agent.planner.executor.execute

    def debug_node_execute(*args, **kwargs):
        print(
            "Debug node execution - request"
        )  # This print statement gives a place to set a breakpoint
        response = original_node_execute(*args, **kwargs)
        print("Debug node execution - response")
        return response

    # Apply instrumented methods
    agent.planner.executor.execute = debug_node_execute

    task = "3 steps draw a digital phoenix using computer emoji characters."
    execution_result = agent.execute(task)

    print(f"Execution Result: {execution_result}")


if __name__ == "__main__":
    main()
