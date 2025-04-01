from typing import Dict, List

from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    default_subscription,
    message_handler, TopicId,
)
from autogen_core.models import (
    AssistantMessage,
    LLMMessage,
    UserMessage,
)

from agent_core.agents import Agent
from message import IntermediateResponse, Request, FinalResponse
class Worker(RoutedAgent):
    def __init__(self, source_agent: Agent, topic_type: str,  max_round: int) -> None:
        super().__init__("A agent worker.")
        self.agent = source_agent
        self._topic_type = topic_type
        self._history: List[LLMMessage] = []
        self._buffer: Dict[int, List[IntermediateResponse]] = {}
        self._round = 0
        self._max_round = max_round

    @message_handler
    async def handle_request(self, message: Request, ctx: MessageContext) -> None:
        print(f"{'-' * 80}\n {self.id.type} received user question")
        self._history.append(UserMessage(content=message.content, source="user"))

        self.agent.background = ", ".join(message.content for message in self._history)
        agent_response = self.agent.execute(message.content)
        assert isinstance(agent_response, str)

        self._history.append(AssistantMessage(content=agent_response, source=self.metadata["type"]))
        print(f"{'-'*80}\n {self.id.type} response on user question:\n{agent_response}")

        print(f"{'-' * 80}\n {self.id.type} dispatches back with answer to SuperVisorAgent")
        self._round += 1
        if self._round == self._max_round:
            await self.publish_message(FinalResponse(answer=agent_response, sender=self.id.type), topic_id=TopicId(type="SuperVisorAgent", source=self.id.type))
        else:
            await self.publish_message(
                IntermediateResponse(
                    content=agent_response,
                    question=message.question,
                    answer=agent_response,
                    round=self._round,
                    sender=self.id.type
                ),
                topic_id=TopicId(type="SuperVisorAgent", source=self.id.type),
            )
