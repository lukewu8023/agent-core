from typing import Optional
from agent_core.agent_basic import AgentBasic
from agent_core.models.model_registry import ModelRegistry


class BaseExecutor(AgentBasic):
    """
    Base executor that handles the actual execution of prompts with the model.
    Provides a simple abstraction over the model.process() call.
    """

    def __init__(
        self, model_name: Optional[str] = None, log_level: Optional[str] = None
    ):
        """Initialize the executor with an optional model name and log level."""
        super().__init__(self.__class__.__name__, model_name, log_level)

    def execute(self, prompt: str, model_name: Optional[str] = None) -> str:
        """
        Execute the given prompt using the specified model (or default model).

        Args:
            prompt: The prompt to send to the model
            model_name: Optional model name to use (overrides the default)

        Returns:
            The model's response as a string
        """
        model_to_use = model_name or self.model_name
        self.logger.debug(f"Executing prompt with model {model_to_use}")
        return ModelRegistry.get_model(model_to_use).process(prompt)
