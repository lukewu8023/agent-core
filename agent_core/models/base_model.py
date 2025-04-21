# models/base_model.py

from abc import ABC, abstractmethod
from agent_core.config import Environment
from agent_core.utils.logger import get_logger


class BaseModel(ABC):

    Environment()

    def __init__(self, input_tokens: int = 0, output_tokens: int = 0):
        self.name = self.name()
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.logger = get_logger(self.__class__.__name__)

    def process(self, request: str) -> (str, int):
        self.logger.debug(f"LLM Request {request}")
        response = self.invoke(request)
        self.logger.debug(f"LLM Response {response}")
        return response

    @abstractmethod
    def invoke(self, request: str):
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    def add_token(self, response):
        if hasattr(response, "usage_metadata"):
            self.output_tokens = self.output_tokens + response.usage_metadata['output_tokens']
            self.input_tokens = self.input_tokens + response.usage_metadata['input_tokens']
