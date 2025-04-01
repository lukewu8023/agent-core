from typing import List

from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    type_subscription,
    message_handler, TopicId,
)

from message import FinalResponse, Question, Request, Answer, IntermediateResponse


@type_subscription("SuperVisorAgent")
class AgentAggregator(RoutedAgent):
    def __init__(self, target_topic: str, second_target_topic: str) -> None:
        super().__init__("SuperVisorAgent")
        self._buffer: List[FinalResponse] = []
        self._target_topic = target_topic
        self._second_target_topic = second_target_topic

    @message_handler
    async def handle_question(self, message: Question, ctx: MessageContext) -> None:
        print(f"{'-'*80}\n {self.id.type} received question:\n{message.content}")
        prompt = (
            f"Please solve the following problem:\n{message.content}\n"
        )
        print(f"{'-'*80}\n{self.id.type} analyze user's question")
        print(f"{'-'*80}\n{self.id.type} dispatches user request to {self._target_topic}")
        await self.publish_message(Request(content=prompt, question=message.content), topic_id=TopicId(self._target_topic, source="SuperVisorAgent"))

    @message_handler
    async def handle_response(self, message: IntermediateResponse, ctx: MessageContext) -> None:
        print(f"{'-'*80}\n {self.id.type} received intermediate answer from {message.sender}")

        print(f"\n {self.id.type} validate intermediate answer")
        print(f"\n {self.id.type} validate pass")

        print(f"{'-' * 80}\n {self.id.type} dispatches user request to {self._second_target_topic}")

        prompt = "These are the solutions to the problem from other agents:\n"
        prompt += f"One agent solution: {message.answer}\n"
        prompt += (
            "Using the solutions from other agents as additional information, "
            "can you provide your answer to the problem? "
            f"The original problem is {message.question}. "
        )
        await self.publish_message(Request(content=prompt, question=message.question), topic_id=TopicId(type=self._second_target_topic, source="SuperVisorAgent"))

    @message_handler
    async def handle_final_solver_response(self, message: FinalResponse, ctx: MessageContext) -> None:
        self._buffer.append(message)
        print(f"{'-'*80}\n {self.id.type} received final from answer {message.sender}")

        print(f"\n {self.id.type} validate final answer")
        print(f"\n {self.id.type} validate pass")

        # Find the majority answer.
        answers = [resp.answer for resp in self._buffer]
        majority_answer = max(set(answers), key=answers.count)
        # Publish the aggregated response.
        await self.publish_message(Answer(content=majority_answer), topic_id=DefaultTopicId())
        # Clear the responses.
        self._buffer.clear()
        print(f"{'-'*80}\n{self.id.type} publishes final answer:\n{majority_answer}")
