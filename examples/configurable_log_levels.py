# examples/configurable_log_levels.py

import sys
import os

# Add the parent directory to sys.path to allow imports from the framework
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from agent_core.agents import Agent
from agent_core.planners import GenericPlanner

# Setting Log Levels in Environment Variable
# export AGENT_CORE_LOG_LEVEL=DEBUG
# python main.py

# Setting Log Levels in Programmatic
# from config.config import Config

# Config.set_log_level("DEBUG")  # globally set to DEBUG


def main():

    agent = Agent(model_name="gemini-1.5-flash-002")
    agent.planner = GenericPlanner(model_name="gemini-1.5-pro-002")
    # agent = Agent(model="gpt-4o-mini", log_level="ERROR")
    # agent.planner = GenericPlanner(model="gpt-3.5-turbo", log_level="DEBUG")

    task = "3 steps draw a flower"
    agent.execute(task)

    execution_result = agent.get_execution_result_summary()
    print(f"Execution Result: {execution_result}")


if __name__ == "__main__":
    main()
