# models/base_model.py

from abc import ABC, abstractmethod

from agent_core.config import Environment

Environment()


class BaseModel(ABC):

    def __init__(self, token: int = 0):
        self.name = self.name()
        self.token = token

    @abstractmethod
    def process(self, command: str) -> (str, int):
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    def add_token(self, response,):
        if (hasattr(response, "usage_metadata")
                and 'total_tokens' in response.usage_metadata):
            self.token = self.token + response.usage_metadata['total_tokens']
