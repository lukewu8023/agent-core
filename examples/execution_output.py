# examples/execution_output.py

import sys
import os

# Add the parent directory to sys.path to allow imports from the framework
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from agent_core.agents import Agent
from agent_core.planners import GenericPlanner


def main():

    agent = Agent()
    agent.planner = GenericPlanner()

    task = "3 steps draw a digital flower using computer characters."
    response = agent.execute(task)
    print(f"Response: {response}")

    execution_responses = agent.execution_responses
    print(f"Execution Response: {agent.execution_responses}")
    execution_history = agent.execution_history
    print(f"Execution History: {execution_history}")
    final_response = agent.get_final_response()
    print(f"Final Result: {final_response}")

    agent.export_execution_trace()


if __name__ == "__main__":
    main()
