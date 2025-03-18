# examples/tools_integration.py

import sys
import os

# Add the parent directory to sys.path to allow imports from the framework
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from agent_core.agents import Agent
from agent_core.planners import GraphPlanner
from langchain_core.tools import tool
from typing import Annotated, List


@tool("event")
def get_event(event_id: Annotated[int, "event id"]) -> dict:
    """Get Event Detail"""
    return {
        "event": {
            "id": 10000,
            "version": 1,
            "date": {"start_time": "1700000000", "end_time": "1700010000"},
        }
    }


@tool("metric")
def get_metric(
    component_name: Annotated[str, "component name"],
    start_time: Annotated[int, "start time"],
    end_time: Annotated[int, "end time"],
) -> List:
    """Get metric data from prometheus by component name"""
    return [
        {
            "component": "IE",
            "name": "CPU Usage",
            "desc": "the cpu usage of the component, unit is %",
            "data": [[123456789, 10], [123456789, 12]],
        },
        {
            "component": "IE",
            "name": "Memory Usage",
            "desc": "the memory usage of the component, unit is %",
            "data": [[123456789, 10], [123456789, 12]],
        },
    ]


@tool("log")
def get_log(
    component_name: Annotated[str, "component name"],
    event_id: Annotated[int, "event id"],
    start_time: Annotated[int, "start time"],
    end_time: Annotated[int, "end time"],
) -> dict:
    """Get log from kibana by component name and event id"""
    return {
        "trace_id": "123456-123456-123456",
        "component": "IE",
        "event_id": "10000",
        "log": "sql execute failed",
    }


@tool("trace")
def get_trace(trace_id: Annotated[str, "trace id"]) -> List:
    """Get trace data by trace id"""
    return [
        {
            "eventId": 10000,
            "traceId": "123456-123456-123456",
            "process": "sql execute failed, no table exist: select * from schema.table",
        }
    ]


def main():
    agent = Agent(model_name="gemini-1.5-flash-002")

    agent.tools = [get_event, get_metric, get_log, get_trace]
    agent.planner = GraphPlanner(model_name="gemini-1.5-pro-002")
    agent.enable_evaluators()

    task = "Find the specifics root cause and get more detail about why the event id: 10000 in IE component failed?"
    agent.execute(task)

    execution_history = agent.execution_history
    print(f"Execution History: {execution_history.execution_history_to_str()}")
    execution_result = agent.get_execution_result_summary()
    print(f"Execution Result: {execution_result}")


if __name__ == "__main__":
    main()
