import base64
import json
import uuid

from agent_core.agents import Agent
from agent_core.planners import GenericPlanner
from common.client import A2ACardResolver
from common.types import (
    AgentCard,
    DataPart,
    Message,
    Part,
    Task,
    TaskSendParams,
    TaskState,
    TextPart,
)
from common import types

from remote_agent_connection import RemoteAgentConnections, TaskUpdateCallback


class SuperVisorAgent:
    """The supervisor agent.

    This is the agent responsible for choosing which remote agents to send
    tasks to and coordinate their work.
    """

    def __init__(
        self,
        remote_server_addresses: list[str],
        task_callback: TaskUpdateCallback | None = None,
    ):
        self.task_callback = task_callback
        self.remote_server_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        for address in remote_server_addresses:
            card_resolver = A2ACardResolver(address)
            card = card_resolver.get_agent_card()
            remote_connection = RemoteAgentConnections(card)
            self.remote_server_connections[card.name] = remote_connection
            self.cards[card.name] = card
        agent_info = []
        for ra in self.list_remote_agents():
            agent_info.append(json.dumps(ra))
        self.agents = '\n'.join(agent_info)

    def register_agent_card(self, card: AgentCard):
        remote_connection = RemoteAgentConnections(card)
        self.remote_server_connections[card.name] = remote_connection
        self.cards[card.name] = card
        agent_info = []
        for ra in self.list_remote_agents():
            agent_info.append(json.dumps(ra))
        self.agents = '\n'.join(agent_info)

    def create_agent(self) -> Agent:
        agent = Agent(model_name="gemini-2.0-flash-001")
        agent.planner = GenericPlanner()
        agent.tools = [self.list_remote_agents, self.send_task]

        agent.context = self.root_instruction
        return agent

    def root_instruction(self) -> str:
        return f"""You are an expert delegator that can delegate the user request to the
                appropriate remote agents.
                
                Discovery:
                - You can use `list_remote_agents` to list the available remote agents you
                can use to delegate the task.
                
                Execution:
                - For actionable tasks, you can use `send_task` to assign tasks to remote agents to perform.
                Be sure to include the remote agent name when you respond to the user.
                
                You can use `check_pending_task_states` to check the states of the pending
                tasks.
                
                Please rely on tools to address the request, and don't make up the response. If you are not sure, please ask the user for more details.
                Focus on the most recent parts of the conversation primarily.
                
                If there is an active agent, send the request to that agent with the update task tool.
                
                Agents:
                {self.agents}

        """

    def list_remote_agents(self):
        """List the available remote agents you can use to delegate the task."""
        if not self.remote_server_connections:
            return []

        remote_agent_info = []
        for card in self.cards.values():
            remote_agent_info.append(
                {'name': card.name, 'description': card.description}
            )
        return remote_agent_info

    async def send_task(
        self, agent_name: str, message: str,
    ):
        """Sends a task either streaming (if supported) or non-streaming.

        This will send a message to the remote agent named agent_name.

        Args:
          agent_name: The name of the agent to send the task to.
          message: The message to send to the agent for the task.

        Yields:
          A dictionary of JSON data.
        """
        if agent_name not in self.remote_server_connections:
            raise ValueError(f'Agent {agent_name} not found')
        client = self.remote_server_connections[agent_name]
        if not client:
            raise ValueError(f'Client not available for {agent_name}')
        taskId = str(uuid.uuid4())
        sessionId = str(uuid.uuid4())
        task: Task
        messageId = ''
        metadata = {}

        if not messageId:
            messageId = str(uuid.uuid4())
        metadata.update(conversation_id=sessionId, message_id=messageId)
        request: TaskSendParams = TaskSendParams(
            id=taskId,
            sessionId=sessionId,
            message=Message(
                role='user',
                parts=[TextPart(text=message)],
                metadata=metadata,
            ),
            acceptedOutputModes=['text', 'text/plain', 'image/png'],
            # pushNotification=None,
            metadata={'conversation_id': sessionId},
        )
        task = await client.send_task(request, self.task_callback)
        # Assume completion unless a state returns that isn't complete

        if task.status.state == TaskState.INPUT_REQUIRED:
        # TODO: Force user input back
            raise ValueError(f'Agent {agent_name} task {task.id} required further input')
        elif task.status.state == TaskState.CANCELED:
            # Open question, should we return some info for cancellation instead
            raise ValueError(f'Agent {agent_name} task {task.id} is cancelled')
        elif task.status.state == TaskState.FAILED:
            # Raise error for failure
            raise ValueError(f'Agent {agent_name} task {task.id} failed')
        response = []
        if task.status.message:
            # Assume the information is in the task message.
            response.extend(
                convert_parts(task.status.message.parts)
            )
        if task.artifacts:
            for artifact in task.artifacts:
                response.extend(convert_parts(artifact.part))
        return response


def convert_parts(parts: list[Part]):
    rval = []
    for p in parts:
        rval.append(convert_part(p))
    return rval


def convert_part(part: Part):
    if part.type == 'text':
        return part.text
    if part.type == 'data':
        return part.data
    if part.type == 'file':
        # Repackage A2A FilePart to google.genai Blob
        # Currently not considering plain text as files
        file_id = part.file.name
        file_bytes = base64.b64decode(part.file.bytes)
        file_part = types.Part(
            inline_data=types.Blob(
                mime_type=part.file.mimeType, data=file_bytes
            )
        )
        return DataPart(data={'artifact-file-id': file_id})
    return f'Unknown type: {part.type}'
