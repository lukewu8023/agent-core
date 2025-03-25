# examples/reasoning_trace.py
import json
import datetime
import sys
import os

# Add the parent directory to sys.path to allow imports from the framework
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from agent_core.agents import Agent
from agent_core.planners import GraphPlanner


def log_file(data):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_name = f"../log/trace_{timestamp}.json"
    os.makedirs(os.path.dirname(trace_name), exist_ok=True)
    reasoning_name = f"../log/reasoning_{timestamp}.json"
    os.makedirs(os.path.dirname(reasoning_name), exist_ok=True)
    with open(trace_name, "w", encoding="utf-8") as f:
        json.dump(data.execution_history.model_dump(), f, indent=4, ensure_ascii=False)
    with open(reasoning_name, "w", encoding="utf-8") as f:
        json.dump(data.get_execution_reasoning(), f, indent=4, ensure_ascii=False)


def main():

    agent = Agent()
    agent.planner = GraphPlanner()
    agent.enable_evaluators()

    task = "3 steps draw a digital phoenix using computer emoji characters."
    agent.execute(task)

    log_file(agent)


if __name__ == "__main__":
    main()
