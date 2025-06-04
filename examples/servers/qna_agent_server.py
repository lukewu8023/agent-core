from langchain_core.tools import tool

from agent_core.agents.a2a_agent import A2AAgent
from agent_core.protocols.a2a.server import A2AServer
from agent_core.protocols.a2a.server.request_handlers import DefaultA2ARequestHandler
from agent_core.protocols.a2a.types import AgentCapabilities, AgentSkill, AgentCard, AgentAuthentication


@tool("emc")
def emc() -> str:
    """Explain what is EMC"""
    return "EMC : Event Master Center"


qna_agent = A2AAgent()
qna_agent.tools = [emc]
qna_agent.enable_evaluators()
capabilities = AgentCapabilities(streaming=False, pushNotifications=False)
qa_skill = AgentSkill(
    id="qna_agent_skill",
    name="QNA agent skill",
    description="Answer user questions based on knowledge",
    tags=["Q & A", "answer question"],
    examples=["What is github?"],
)
qna_agent_card = AgentCard(
    name="QNA Agent",
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
    agent_card=qna_agent_card,
    request_handler=DefaultA2ARequestHandler(
        agent_executor=qna_agent
    )
)

serverA.start(host='localhost', port=8881)
