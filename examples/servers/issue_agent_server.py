from agent_core.agents.a2a_agent import A2AAgent
from agent_core.protocols.a2a.server import A2AServer
from agent_core.protocols.a2a.server.request_handlers import DefaultA2ARequestHandler
from agent_core.protocols.a2a.types import AgentCapabilities, AgentSkill, AgentCard, AgentAuthentication
from agent_core.protocols.mcp.mcp_server import MCPServer
from agent_core.planners import GraphPlanner


mcp = MCPServer('http://0.0.0.0:8000/mcp')
issue_agent = A2AAgent()
issue_agent.mcp_servers = [mcp]
issue_agent.planner = GraphPlanner(model_name="gemini-1.5-pro-002")
issue_agent.enable_evaluators()

capabilities = AgentCapabilities(streaming=False, pushNotifications=False)

issue_agent_skill = AgentSkill(
    id="issue_agent_skill",
    name="Issue agent skill",
    description="look up information for issue troubleshooting",
    tags=["troubleshooting"],
    examples=["Why cpu usage is high?"],
)
issue_agent_card = AgentCard(
    name="Issue Agent",
    description="Helps with issue troubleshooting",
    url=f"http://localhost:8882",
    version="1.0.0",
    defaultInputModes=["text", "text/plain"],
    defaultOutputModes=["text", "text/plain"],
    capabilities=capabilities,
    skills=[issue_agent_skill],
    authentication=AgentAuthentication(schemes=['public'])
)

serverB = A2AServer(
    agent_card=issue_agent_card,
    request_handler=DefaultA2ARequestHandler(
        agent_executor=issue_agent
    )
)

serverB.start(host="localhost", port=8882)

