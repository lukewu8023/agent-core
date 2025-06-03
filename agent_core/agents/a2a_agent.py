import asyncio
from typing import Optional
from uuid import uuid4

from agent_core.agents import Agent
from agent_core.protocols.a2a.utils.helpers import (
    update_task_with_agent_response, update_task_with_agent_reasoning,
)
from typing_extensions import override

from agent_core.protocols.a2a.server.agent_execution import BaseAgentExecutor
from agent_core.protocols.a2a.server.events.event_queue import EventQueue
from agent_core.protocols.a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Task,
    TextPart, TaskState, TaskStatus,
)
from agent_core.protocols.a2a.utils import create_task_obj


def _get_user_query(task_send_params: MessageSendParams) -> str:
    """Helper to get user query from task send params."""
    part = task_send_params.message.parts[0].root
    if not isinstance(part, TextPart):
        raise ValueError('Only text parts are supported')
    return part.text


class A2AAgent(Agent, BaseAgentExecutor):

    def __init__(self, model_name: Optional[str] = None, log_level: str = "INFO"):
        super().__init__(model_name, log_level)

    @override
    async def on_message_send(
        self,
        request: SendMessageRequest,
        event_queue: EventQueue,
        task: Task | None,
    ) -> None:
        params: MessageSendParams = request.params
        query = _get_user_query(params)

        if not task:
            task = create_task_obj(params)

        agent_response = asyncio.create_task(self.execute(
            query
        ))

        while not agent_response.done():
            print(f"Reasoning : {self.get_execution_reasoning()}")
            await asyncio.sleep(1)

        result = await agent_response
        update_task_with_agent_response(task, result)
        event_queue.enqueue_event(task)

    @override
    async def on_message_stream(
        self,
        request: SendStreamingMessageRequest,
        event_queue: EventQueue,
        task: Task | None,
    ) -> None:
        params: MessageSendParams = request.params
        query = _get_user_query(params)

        if not task:
            task = create_task_obj(params)

        agent_response = asyncio.create_task(self.execute(
            query
        ))

        reason_index = 0

        while not agent_response.done():
            current_reasoning = self.get_execution_reasoning()
            if current_reasoning and len(current_reasoning) != reason_index:
                reason_index = len(current_reasoning)
                intermediate_task = create_task_obj(params)
                update_task_with_agent_reasoning(intermediate_task, current_reasoning[reason_index-1:])
                event_queue.enqueue_event(intermediate_task)
            await asyncio.sleep(1)

        result = await agent_response
        update_task_with_agent_response(task, result)
        event_queue.enqueue_event(task)
