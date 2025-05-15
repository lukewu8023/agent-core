from agent_core.agents.supervisor_agent import SuperVisorAgent
from agent_core.protocols.a2a.server import A2AServer
from agent_core.protocols.a2a.server.request_handlers import DefaultA2ARequestHandler
from agent_core.protocols.a2a.types import AgentCapabilities, AgentSkill, AgentCard, AgentAuthentication


supervisor_agent = SuperVisorAgent(['http://localhost:8881', 'http://localhost:8882'])
capabilities = AgentCapabilities(streaming=False, pushNotifications=False)
supervisor_skill = AgentSkill(
    id="supervisor_agent_skill",
    name="supervisor agent skill",
    description="Answer user questions based on knowledge. Look up information for issue troubleshooting",
    tags=["Q & A", "answer question", "troubleshooting"],
    examples=["What is github?", "Why cpu usage is high?"],
)
supervisor_agent_card = AgentCard(
    name="supervisor Agent",
    description="Helps with answer user query based on knowledge and issue troubleshooting",
    url=f"http://localhost:8880/",
    version="1.0.0",
    defaultInputModes=["text", "text/plain"],
    defaultOutputModes=["text", "text/plain"],
    capabilities=capabilities,
    skills=[supervisor_skill],
    authentication=AgentAuthentication(schemes=['public'])
)

supervisor_agent_server = A2AServer(
    agent_card=supervisor_agent_card,
    request_handler=DefaultA2ARequestHandler(
        agent_executor=supervisor_agent
    )
)

supervisor_agent_server.start(host='localhost', port=8880)
