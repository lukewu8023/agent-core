from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool
from langgraph.graph import add_messages
from typing_extensions import TypedDict, Annotated

from agent_core.agents import Agent
from agent_core.evaluators import BaseEvaluator
from agent_core.planners.base_planner import BasePlanner


class AgentState(TypedDict):
    messages: Optional[Annotated[list, add_messages]]
    planner: Optional[BasePlanner]
    tools: Optional[List[BaseTool]]
    knowledge: Optional[str]
    background: Optional[str]
    model_name: Optional[str]
    log_level: Optional[str]
    evaluators_enabled: Optional[bool]
    evaluators: Optional[Dict[str, BaseEvaluator]]
    task_function: Optional[Any]
    response_function: Optional[Any]


def process_state_schema(agent, state_schema):
    if "planner" in state_schema:
        agent.planner = state_schema["planner"]
    if "tools" in state_schema:
        agent.tools = state_schema["tools"]
    if "knowledge" in state_schema:
        agent.knowledge = state_schema["knowledge"]
    if "background" in state_schema:
        agent.background = state_schema["background"]
    if "evaluators_enabled" in state_schema:
        agent.evaluators_enabled = state_schema["evaluators_enabled"]
    if "evaluators" in state_schema:
        agent.evaluators = state_schema["evaluators"]


def get_task(state_schema):
    if "task_function" in state_schema:
        return state_schema["task_function"](state_schema)
    else:
        if "messages" in state_schema:
            if isinstance(state_schema["messages"], list):
                return state_schema["messages"][-1]["content"]


def set_response(state_schema, response):
    if "response_function" in state_schema:
        return state_schema["response_function"](state_schema, response)
    else:
        if "messages" in state_schema:
            if isinstance(state_schema["messages"], list):
                return {"messages": response}


def agent_execute(state_schema: Optional[Any]):
    model_name = None
    log_level = None
    if "model_name" in state_schema:
        model_name = state_schema["model_name"]
    if "log_level" in state_schema:
        log_level = state_schema["log_level"]
    agent = Agent(model_name, log_level)
    process_state_schema(agent, state_schema)
    response = agent.execute(get_task(state_schema))
    return set_response(state_schema, response)