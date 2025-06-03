import asyncio
import json
import uuid
from typing import Optional, Any
from typing_extensions import override

import nest_asyncio
import httpx
from starlette.websockets import WebSocket

from agent_core.agents import Agent
from agent_core.agents.a2a_agent import A2AAgent
from agent_core.protocols.a2a.client import A2ACardResolver, A2AClient
from agent_core.protocols.a2a.types import (
    AgentCard, SendMessageResponse, SendMessageSuccessResponse, GetTaskResponse, Task, SendStreamingMessageResponse,
    GetTaskRequest, TaskQueryParams, TaskState,
)
from pydantic import BaseModel, Field

from agent_core.utils.llm_chat import LLMChat

ROUTE_PROMPT = """
Given a raw text task to a language model select the agent best suited for \
the task. You will be given the names of the available agents and a description of \
what the agent is best suited for. You may also revise the original task if you \
think that revising it will ultimately lead to a better response from the language \
model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{
    "agent": string \\ name of the prompt to use
    "reason": string \\ a reason why choose this agent to handle this task
}}
```

REMEMBER: "agent" MUST be one of the agent names specified below.

<< Agents Information >>
{agents}

<< TASK >>
{task}

<< OUTPUT (must include ```json at the start of the response) >>
<< OUTPUT (must end with ```) >>
"""


def run_coro_sync(coro):
    """
    Runs an async coroutine from sync context, even within a running loop (e.g. Jupyter).
    Handles return values and exceptions correctly.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # For environments like Jupyter, use nest_asyncio to allow re-entry
        nest_asyncio.apply()

        future = asyncio.ensure_future(coro)

        # run until the future is done
        async def wait_for_result():
            try:
                return await future
            except Exception as e:
                raise e  # re-raise exception into outer sync world

        return loop.run_until_complete(wait_for_result())
    else:
        return loop.run_until_complete(coro)


def create_send_message_payload(
    text: str, task_id: str | None = None, context_id: str | None = None
) -> dict[str, Any]:
    """Helper function to create the payload for sending a task."""
    payload: dict[str, Any] = {
        'message': {
            'role': 'user',
            'parts': [{'type': 'text', 'text': text}],
            'messageId': uuid.uuid4().hex,
        },
    }

    if task_id:
        payload['message']['taskId'] = task_id

    if context_id:
        payload['message']['contextId'] = context_id
    return payload


def return_response(response: Any) -> None:
    """Helper function to print the JSON representation of a response."""
    if hasattr(response, 'root'):
        return response.root.model_dump_json(exclude_none=True)
    else:
        return response.model_dump(mode="json", exclude_none=True)


async def send_task(client: A2AClient, payload: dict[str, Any]) -> None:
    send_response: SendMessageResponse = await client.send_message(
        payload=payload
    )
    return return_response(send_response)


async def send_task_stream(client: A2AClient, payload: dict[str, Any], ws: WebSocket) -> None:
    async for response in client.send_message_streaming(payload=payload):
        task = response.root
        result: Task = getattr(task, "result", None)
        for part in result.artifacts[0].parts:
            await ws.send_text(part.root.model_dump_json(exclude_none=True))


class TaskSchema(BaseModel):
    agent_name: str = Field(..., description="The name of the agent to send the task to.")
    message: str = Field(..., description="The message to send to the agent for the task")


async def load_agent(address: str) -> AgentCard:
    async with httpx.AsyncClient() as client:
        card_resolver = A2ACardResolver(httpx_client=client, base_url=address)
        card = await card_resolver.get_agent_card()
        return card


class SuperVisorAgent(A2AAgent):
    """The supervisor agent.

    This is the agent responsible for choosing which remote agents to send
    tasks to and coordinate their work.
    """

    def __init__(self, remote_server_addresses: list[str], model_name: Optional[str] = None):
        super().__init__(model_name)
        self.cards: dict[str, AgentCard] = {}
        self.remote_server_addresses = remote_server_addresses

    def register_agent_card(self, card: AgentCard):
        self.cards[card.name] = card

    def get_agents_info(self):
        remote_agent_info = []
        for card in self.cards.values():
            info_dict = dict()
            info_dict['name'] = card.name
            info_dict['description'] = card.description
            info_dict['skill'] = []
            for skill in card.skills:
                skill_dict = dict()
                skill_dict['name'] = skill.name
                skill_dict['description'] = skill.description
                skill_dict['tags'] = skill.tags
                skill_dict['examples'] = skill.examples
                info_dict['skill'].append(skill_dict)
            remote_agent_info.append(info_dict)
        return remote_agent_info

    async def init(self):
        for address in self.remote_server_addresses:
            card = await load_agent(address)
            self.cards[card.name] = card

    async def route(self, task: str):
        if len(self.cards) == 0 and len(self.remote_server_addresses) != 0:
            await self.init()
        routing_prompt = ROUTE_PROMPT.format(agents=json.dumps(self.get_agents_info()), task=task)
        routing = await LLMChat().process(routing_prompt)
        response = json.loads(routing.replace("```json", '').replace("```", ''))
        agent = response['agent']
        return agent

    @override
    async def execute(self, task: str):
        agent = await self.route(task)
        card = self.cards[agent]
        payload = create_send_message_payload(text=task)
        timeout = httpx.Timeout(connect=3600.0, read=3600.0, write=3600.0, pool=3600.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            a2a_client = await A2AClient.get_client_from_agent_card_url(client, card.url)
            return await send_task(a2a_client, payload)

    async def execute_ws(self, task: str, ws: WebSocket):
        agent = await self.route(task)
        card = self.cards[agent]
        payload = create_send_message_payload(text=task)
        timeout = httpx.Timeout(connect=3600.0, read=3600.0, write=3600.0, pool=3600.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            a2a_client = await A2AClient.get_client_from_agent_card_url(client, card.url)
            await send_task_stream(a2a_client, payload, ws)