# models/base_model.py
import asyncio
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent_core.config import Environment
from agent_core.utils.logger import get_logger


class BaseModel(ABC):

    Environment()

    model_instance: ChatOpenAI

    def __init__(self, input_tokens: int = 0, output_tokens: int = 0):
        self.name = self.name()
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.logger = get_logger(self.__class__.__name__)

    async def process(self, request: str) -> (str, int):
        self.logger.debug(f"LLM Request {request}")
        response = await self.invoke(request)
        self.logger.debug(f"LLM Response {response}")
        return response

    async def invoke(self, request: str) -> str:
        return await asyncio.to_thread(self.invoke_sync, request)

    def invoke_sync(self, request: str) -> str:
        messages = [
            HumanMessage(request),
        ]
        response = self.model_instance.invoke(messages)
        # Extract the 'content' attribute to return a string
        if hasattr(response, "content"):
            self.add_token(response)
            return response.content
        else:
            # Fallback in case 'content' is missing
            return str(response)

    @abstractmethod
    def name(self) -> str:
        pass

    def add_token(self, response):
        if hasattr(response, "usage_metadata"):
            self.output_tokens = self.output_tokens + response.usage_metadata['output_tokens']
            self.input_tokens = self.input_tokens + response.usage_metadata['input_tokens']
