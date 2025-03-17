# models/base_model.py

from abc import ABC, abstractmethod
from agent_core.config import Environment
from agent_core.utils.logger import get_logger

Environment()


class BaseModel(ABC):

    def __init__(self, token: int = 0):
        self.name = self.name()
        self.token = token
        self.logger = get_logger(self.__class__.__name__)

    def process(self, request: str) -> (str, int):
        self.logger.info(f"LLM Request {request}")
        response = self.invoke(request)
        self.logger.info(f"LLM Response {response}")
        return response

    @abstractmethod
    def invoke(self, request: str):
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    def add_token(self, response):
        if (hasattr(response, "usage_metadata")
                and 'total_tokens' in response.usage_metadata):
            self.token = self.token + response.usage_metadata['total_tokens']
