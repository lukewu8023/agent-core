from typing import Any, Optional, List, Dict, Sequence
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import ChatMessage, TextMessage
from agent_core.agents import Agent
from agent_core.planners.base_planner import BasePlanner
from agent_core.evaluators import BaseEvaluator
from langchain_core.tools import BaseTool
import asyncio
from autogen_agentchat.base import Response

class PhaenixAgent(BaseChatAgent):
    def __init__(
            self,
            name: str,
            model_name: Optional[str] = None,
            log_level: str = "INFO",
            tools: List[BaseTool] = None,
            planner: Optional[BasePlanner] = None,
            knowledge: Optional[str] = None,
            background: Optional[str] = None,
            evaluators_enabled: bool = False,
            evaluators: Dict[str, BaseEvaluator] = None,
            autogen_tools: List[Any] = None,
            **kwargs
    ):
        super().__init__(name=name, description="Phaenix Integrate with Autogen")

        self.core_agent = Agent(
            model_name=model_name,
            log_level=log_level
        )

        if planner is not None:
            self.core_agent.planner = planner
        if tools is not None:
            self.core_agent.tools = tools
        if knowledge is not None:
            self.core_agent.knowledge = knowledge
        if background is not None:
            self.core_agent.background = background
        if evaluators_enabled is not None:
            self.core_agent.evaluators_enabled = evaluators_enabled
        if evaluators is not None:
            self.core_agent.evaluators = evaluators

        if autogen_tools is not None:
            self._integrate_autogen_tools(autogen_tools)

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage,)
    
    def _integrate_autogen_tools(self, tools: List[Any]):
        class AutogenToolAdapter(BaseTool):
            def __init__(self, tool):
                super().__init__(
                    name=tool.name,
                    description=tool.description,
                    args_schema=tool.args_schema
                )
                self.native_tool = tool
            def _run(self, *args, **kwargs):
                return self.native_tool.execute(*args, **kwargs)

        self.core_agent.tools += [AutogenToolAdapter(t) for t in tools]
    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: Any) -> Response:

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.core_agent.execute(messages[0].content)
        )

        return Response(
            chat_message=TextMessage(
                content=str(result),
                source=self.name
            )
        )

    async def on_reset(self, cancellation_token: Any) -> None:
        pass
