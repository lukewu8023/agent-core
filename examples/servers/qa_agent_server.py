from agent_core.agents.a2a_agent import A2AAgent
from agent_core.protocols.a2a.server import A2AServer
from agent_core.protocols.a2a.server.request_handlers import DefaultA2ARequestHandler
from agent_core.protocols.a2a.types import AgentCapabilities, AgentSkill, AgentCard, AgentAuthentication


qa_agent = A2AAgent()
qa_agent.background = "EMC : Event Master Center"
capabilities = AgentCapabilities(streaming=False, pushNotifications=False)
qa_skill = AgentSkill(
    id="qa_agent_skill",
    name="QA agent skill",
    description="Answer user questions based on knowledge",
    tags=["Q & A", "answer question"],
    examples=["What is github?"],
)
qa_agent_card = AgentCard(
    name="QA Agent",
    description="Helps with answer user query based on knowledge",
    url=f"http://localhost:8881/",
    version="1.0.0",
    defaultInputModes=["text", "text/plain"],
    defaultOutputModes=["text", "text/plain"],
    capabilities=capabilities,
    skills=[qa_skill],
    authentication=AgentAuthentication(schemes=['public'])
)

serverA = A2AServer(
    agent_card=qa_agent_card,
    request_handler=DefaultA2ARequestHandler(
        agent_executor=qa_agent
    )
)

serverA.start(host='localhost', port=8881)
