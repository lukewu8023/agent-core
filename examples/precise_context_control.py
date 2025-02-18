# examples/precise_context_control.py

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
    agent.planner = GenericPlanner(model_name="gemini-1.5-pro-002")

    context = agent.context
    print(context)

    context.add_context(
        "role",
        f"""
        you are an digital artist, able to use computer character to draw digital picture.
        """,
    )
    print(context)

    task = "draw a flower"
    agent.execute(task)

    execution_result = agent.get_execution_result_summary()
    print(f"Execution Result: {execution_result}")


if __name__ == "__main__":
    main()
