# examples/mcp_tools_integration.py
import asyncio

from langchain_core.tools import tool

from agent_core.agents.a2a_agent import A2AAgent
from agent_core.protocols.mcp.mcp_server import MCPServer
from agent_core.planners import GraphPlanner


@tool("hello")
def hello(name: str) -> str:
    """Hello World"""
    return 'hello ' + name


async def main():
    mcp = MCPServer('http://0.0.0.0:8000/mcp')
    agent = A2AAgent()
    agent.mcp_servers = [mcp]
    agent.planner = GraphPlanner(model_name="gemini-1.5-pro-002")
    agent.enable_evaluators()

    task = "Find the specifics root cause and get more detail about why the event id: 10000 in IE component failed?"
    execution_result = asyncio.create_task(agent.execute(task))

    while not execution_result.done():
        print(f"Reasoning : {agent.get_execution_reasoning()}")
        await asyncio.sleep(3)

    result = await execution_result
    print(f"Final Execution Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
