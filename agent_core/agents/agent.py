# agents/agent.py

from typing import Optional, List

from langchain_core.tools import BaseTool
from agent_core.agent_basic import AgentBasic
from agent_core.entities.steps import Steps, Step
from agent_core.models.model_registry import ModelRegistry
from agent_core.planners.base_planner import BasePlanner
from agent_core.utils.context_manager import ContextManager
from agent_core.evaluators.evaluators import get_evaluator
from agent_core.evaluators import BaseEvaluator


class Agent(AgentBasic):
    """
    The Agent coordinates task execution with or without a Planner.
    It now exposes two prompts:
      - execute_prompt: overrides how we generate the no-planner prompt
      - summary_prompt (used in get_execution_result)
    """

    DEFAULT_EXECUTE_PROMPT = """
{context_section}

**Background**
{background}

**Task**
{task}
    """

    DEFAULT_SUMMARY_PROMPT = """
You are an assistant summarizing the outcome of a multi-step plan execution.
Below is the complete step-by-step execution history. Provide a well-structured summary describing how the solution was achieved and any notable details, make sure to include each step's result in the final summary. 

**Execution History**
{history_text}

**Output format**
{{
    "summary": "A detailed summary of how the solution was achieved",
    "output_result": "The final output/result of the execution",
    "conclusion": "A brief conclusion about the overall execution"
}}

**Important**
- Ensure your response is valid JSON without any additional text or comments (// explain).
    """

    DEFAULT_FINAL_RESPONSE_PROMPT = """
You are an assistant to response user's query.
Given user'query and step-by-step result of execution history. 
Generate the final response to user. The final answer usually in the last step.

**User Query**
{task}

**Execution History**
{history_text}

Response:
    """

    def __init__(
        self, model_name: Optional[str] = None, log_level: Optional[str] = None
    ):
        """
        If 'model' is not provided, the default model from config will be used.
        'log_level' can override the framework-wide default for this Agent specifically.
        """
        # Initialize and load models
        ModelRegistry.load_models()

        # This list holds execution data for each step in sequence.
        super().__init__(self.__class__.__name__, model_name, log_level)
        self._execution_history: Steps = Steps()

        self.planner = None
        self.tools: Optional[List[BaseTool]] = None

        # Default knowledge / background
        self.knowledge = ""  # Used to guide how we make plans
        self.background = ""  # Used during execution steps

        # The context manager (use get_context())
        self.context = ContextManager()

        # Prompt strings for direct (no-planner) usage and summary
        self.execute_prompt = self.DEFAULT_EXECUTE_PROMPT
        self.summary_prompt = self.DEFAULT_SUMMARY_PROMPT
        self.response_prompt = self.DEFAULT_FINAL_RESPONSE_PROMPT

        # NEW: evaluator management
        self.evaluators_enabled = False
        self.evaluators = {}
        self._load_default_evaluators()

        self.logger.info("Agent instance is created.")

    def execute(self, task: str):
        """
        1) If no planner, do direct single-step with the model (use background).
        2) If planner, plan(...) -> then call execute_plan(...).
        """
        self.logger.info(f"Agent is executing task: {task}")

        # Case 1: No planner => direct single-step
        if not self.planner:
            return self.execute_without_planner(task)

        # Case 2: Using a planner => first create steps/graph
        current_categories = list(self.evaluators.keys())
        plan = self.planner.plan(
            task=task,
            tools=self.tools,
            knowledge=self.knowledge,
            background=self.background,
            categories=current_categories,
        )

        # Now just call planner's execute_plan(...) in a unified way
        self._execution_history = self.planner.execute_plan(
            task=task,
            plan=plan,
            context_manager=self.context,
            background=self.background,
            evaluators_enabled=self.evaluators_enabled,
            evaluators=self.evaluators,
        )
        self._execution_history.token = self.get_token()
        return self._execution_history.get_last_step_output()

    def execute_without_planner(self, task: str):
        context_section = self.context.context_to_str()
        final_prompt = self.execute_prompt.format(
            context_section=context_section,
            background=self.background,
            task=task,
        )
        response = self._model.process(final_prompt)
        self.logger.info(f"Response: {response}")
        self._execution_history.add_step(
            Step(
                name="Direct Task Execution",
                description=task,
                result=str(response),
                prompt=final_prompt,
            )
        )
        return response

    def get_token(self):
        return ModelRegistry.get_token()

    def get_final_response(self, task: str) -> str:
        history_text = self._execution_history.execution_history_to_str()
        final_response_prompt = self.response_prompt.format(
            task=task, history_text=history_text
        )
        self.logger.info("Generating final response.")
        final_response = self._model.process(final_response_prompt)
        return str(final_response)

    def get_execution_result_summary(self) -> str:
        """
        Produce an overall summary describing how the solution was completed,
        using the LLM (agent's model) to format the final explanation if desired.
        """
        if not self._execution_history:
            return (
                "No direct step-based execution history recorded. "
                "(If you used GraphPlanner, the node-based execution is stored inside the planner.)"
            )

        history_text = self._execution_history.execution_history_to_str()
        final_prompt = self.summary_prompt.format(history_text=history_text)

        self.logger.info("Generating execution result summary.")
        summary_response = self._model.process(final_prompt)
        cleaned = summary_response.replace("```json", "").replace("```", "").strip()
        return str(cleaned)

    def planner(self, planner):
        if not issubclass(planner.__class__, BasePlanner):
            error_msg = "Planner must be an instance of BasePlanner."
            self.logger.error(error_msg)
            raise TypeError(error_msg)
        self.planner = planner
        self.logger.info(f"Agent planner uses model: {planner.__class__.__name__}")

    @property
    def execution_history(self) -> Steps:
        """
        Read-only access to the execution history.
        Each item is a dict with keys: 'step_name', 'step_description', 'step_result'.
        """
        return self._execution_history

    def enable_evaluators(self):
        self.evaluators_enabled = True
        self.logger.info("Evaluators have been enabled.")

    def disable_evaluators(self):
        self.evaluators_enabled = False
        self.logger.info("Evaluators have been disabled.")

    @property
    def execution_responses(self) -> str:
        """
        Read-only access to the execution responses.
        Combine all 'step_result' together.
        """
        return self._execution_history.execution_history_to_responses()

    def _load_default_evaluators(self):
        """
        Load a default mapping of category -> evaluator (all referencing the current model).
        Make a local copy so user modifications won't affect the original file.
        """
        evaluators = get_evaluator(self.model_name)
        self.evaluators = dict(evaluators)

    def add_evaluator(self, category: str, evaluator: BaseEvaluator):
        """
        Insert or override a evaluator for the given category.
        """
        self.evaluators[category] = evaluator

    def update_evaluator(self, category: str, evaluator: BaseEvaluator):
        """
        Update the evaluator for an existing category.
        If the category doesn't exist, we log a warning and add it.
        """
        if category in self.evaluators.keys():
            self.evaluators[category] = evaluator
        else:
            self.logger.warning(
                f"Category '{category}' not found in evaluator. Creating new entry."
            )
            self.evaluators[category] = evaluator
