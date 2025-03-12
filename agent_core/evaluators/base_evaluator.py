# evaluators/base_evaluator.py

from abc import abstractmethod
from typing import Optional

from agent_core.agent_basic import AgentBasic
from agent_core.evaluators.entities.evaluator_result import EvaluatorResult
from agent_core.utils.context_manager import ContextManager


class BaseEvaluator(AgentBasic):
    """
    A base class for all evaluator. Every evaluator must implement `evaluator()`.
    """

    def __init__(self, model_name: Optional[str] = None, log_level: Optional[str] = None,
                 evaluation_threshold: Optional[float] = 0.8, max_attempt: Optional[int] = 3):
        """
        Pass in the agent's model instance so we can call model.process(...) for evaluation prompts.
        Optionally specify log_level for debug or other logs.
        'prompt' can override the default prompt template.
        """
        super().__init__(self.__class__.__name__, model_name, log_level)
        self.evaluation_threshold = evaluation_threshold
        self.prompt = self.default_prompt()
        self.max_attempt = max_attempt
        self.name = self.__class__.__name__

    @abstractmethod
    def default_prompt(self):
        pass

    @abstractmethod
    def evaluate(self, root_task: str, request: str, response: str, background: str, context_manager: ContextManager) -> EvaluatorResult:
        """
        Perform evaluator on the given request and response.
        """
        pass

