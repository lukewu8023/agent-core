# examples/knowledge_background.py

import sys
import os

# Add the parent directory to sys.path to allow imports from the framework
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from agent_core.agents import Agent
from agent_core.planners import GenericPlanner


def main():

    agent = Agent("gemini-1.5-flash-002")
    agent.planner = GenericPlanner(model_name="gemini-1.5-pro-002")

    agent.knowledge = "to draw a object you need to take 3 steps, 1) prepare tools, 2) prepare paper, 3) draw the object"
    agent.background = "You are a professional artist"

    task = "draw a flower"
    agent.execute(task)


if __name__ == "__main__":
    main()
