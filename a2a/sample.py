from agent_core.agents import Agent
from common.server import A2AServer
from common.server.agent_task_manager import AgentTaskManager
from common.types import AgentCapabilities, AgentSkill, AgentCard
from supervisor_agent import SuperVisorAgent

class QAAgent:

    def __init__(self):
        self.agent = Agent()

    def invoke(self, query) -> str:
        response = self.agent.execute(query)
        return response

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

class IssueAgent:

    def __init__(self):
        self.agent = Agent()

    def invoke(self, query) -> str:
        response = self.agent.execute(query)
        return response

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

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
)

issue_agent_skill = AgentSkill(
    id="issue_agent_skill",
    name="Issue agent skill",
    description="look up information from elk for issue troubleshooting",
    tags=["look up information", "troubleshooting"],
    examples=["Why cpu usage is high?"],
)
issue_agent_card = AgentCard(
    name="Issue Agent",
    description="Helps with issue troubleshooting",
    url=f"http://localhost:8881/",
    version="1.0.0",
    defaultInputModes=["text", "text/plain"],
    defaultOutputModes=["text", "text/plain"],
    capabilities=capabilities,
    skills=[issue_agent_skill],
)

serverA = A2AServer(
    agent_card=qa_agent_card,
    task_manager=AgentTaskManager(QAAgent()),
    host="localhost",
    port=8881
)

serverB = A2AServer(
    agent_card=issue_agent_card,
    task_manager=AgentTaskManager(IssueAgent()),
    host="localhost",
    port=8882
)

serverA.start()  # http://localhost:8881/.well-known/agent.json
serverB.start()  # http://localhost:8882/.well-known/agent.json

# supervisor_agent = SuperVisorAgent(['http://localhost:8881', 'http://localhost:8882']).create_agent()
