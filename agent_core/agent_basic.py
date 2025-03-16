from abc import ABC
from typing import Optional
from agent_core.models.base_model import BaseModel
from agent_core.models.model_registry import ModelRegistry
import os
from agent_core.utils.logger import get_logger


class AgentBasic(ABC):

    def __init__(
        self, name, model_name: Optional[str] = None, log_level: Optional[str] = None
    ):
        """
        If 'model' is not provided, the default model from config will be used.
        'log_level' can override the framework-wide default for this Agent specifically.
        """
        self.name = name
        self._model: Optional[BaseModel] = None
        self._model_name: Optional[str] = None
        self.logger = get_logger(self.__class__.__name__, log_level)
        self.model_name = model_name if model_name else os.getenv("DEFAULT_MODEL")

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, model_name: str):
        self._model_name = model_name
        self._model = ModelRegistry.get_model(model_name)
        self.logger.info(f"{self.name} uses model: {model_name}")
