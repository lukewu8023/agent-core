# examples/mcp_tools_integration.py
import asyncio

from langchain_core.tools import tool

from agent_core.agents import Agent
from agent_core.agents.a2a_agent import A2AAgent
from agent_core.protocols.mcp.mcp_server import MCPServer
from agent_core.planners import GraphPlanner


@tool("emc")
def emc() -> str:
    """Explain what is EMC"""
    return "EMC : Event Master Center"


async def main():
    mcp = MCPServer('http://0.0.0.0:8000/mcp')
    agent = Agent()
    agent.mcp_servers = [mcp]
    agent.enable_evaluators()

    task = "What is EMC?"
    execution_result = asyncio.create_task(agent.execute(task))

    while not execution_result.done():
        print(f"Reasoning : {agent.get_execution_reasoning()}")
        await asyncio.sleep(3)

    result = await execution_result
    print(f"Final Execution Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
