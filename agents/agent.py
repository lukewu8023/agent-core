# agents/agent.py

from typing import Optional, List

from planners.generic_planner import GenericPlanner
from planners.graph_planner import GraphPlanner
from models.model_registry import ModelRegistry
from utils.logger import get_logger
from utils.context_manager import get_context
from config import Config


class Agent:
    """
    The Agent coordinates task execution with or without a Planner.
    It now exposes two prompts:
      - execute_prompt: overrides how we generate the no-planner prompt
      - summary_prompt (used in get_execution_result)
    """

    DEFAULT_EXECUTE_PROMPT = """\
{context_section}
Background: {background}
Task: {task}
"""

    DEFAULT_SUMMARY_PROMPT = """\
You are an assistant summarizing the outcome of a multi-step plan execution.
Below is the complete step-by-step execution history. Provide a concise,
well-structured summary describing how the solution was achieved and any
notable details. Include each step's role in the final outcome.

Execution History:
{history_text}

Summary:
"""

    def __init__(self, model: Optional[str] = None, log_level: Optional[str] = None):
        """
        If 'model' is not provided, the default model from config will be used.
        'log_level' can override the framework-wide default for this Agent specifically.
        """
        self.logger = get_logger("agent", log_level)
        self._model = None
        self._planner = None
        self.tools = None
        # This list holds execution data for each step in sequence.
        # Example entry:
        # {
        #   "step_name": "Draw the stem",
        #   "step_description": "Draw a vertical line as the flower's stem",
        #   "step_result": "Stem drawn successfully."
        # }
        self._execution_history = []

        # Default knowledge / background
        self.knowledge = ""  # Used to guide how we make plans
        self.background = ""  # Used during execution steps

        # The context manager (use get_context())
        self._context = get_context()

        # Prompt strings for direct (no-planner) usage and summary
        self._execute_prompt = self.DEFAULT_EXECUTE_PROMPT
        self._summary_prompt = self.DEFAULT_SUMMARY_PROMPT

        if not model:
            model = Config.DEFAULT_MODEL

        # Use the property setter to initialize the model
        self.model = model

        self.logger.info("Agent instance created.")

    @property
    def context(self):
        """
        Expose the agent's context manager so we can do:
          c = agent.context
          c.add_context("role", "...")
        """
        return self._context

    @property
    def execute_prompt(self) -> str:
        """Prompt used when no planner is set (single-step)."""
        return self._execute_prompt

    @execute_prompt.setter
    def execute_prompt(self, value: str):
        self._execute_prompt = value

    @property
    def summary_prompt(self) -> str:
        """Prompt used for final summarizing of execution history."""
        return self._summary_prompt

    @summary_prompt.setter
    def summary_prompt(self, value: str):
        self._summary_prompt = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model_name: str):
        model = ModelRegistry.get_model(model_name)
        if not model:
            self.logger.error(f"Model '{model_name}' not found in registry.")
            raise ValueError(f"Model '{model_name}' is not supported.")
        self._model = model
        self.logger.info(f"Agent model set to: {model.name}")

    @property
    def planner(self) -> Optional[GenericPlanner]:
        return self._planner

    @planner.setter
    def planner(self, planner: GenericPlanner):
        if not isinstance(planner, GenericPlanner):
            self.logger.error("Planner must be an instance of GenericPlanner.")
            raise TypeError("Planner must be an instance of GenericPlanner.")
        self._planner = planner
        self.logger.info(f"Agent planner set to: {planner.__class__.__name__}")

    @property
    def execution_history(self) -> List[dict]:
        """
        Read-only access to the execution history.
        Each item is a dict with keys: 'step_name', 'step_description', 'step_result'.
        """
        return self._execution_history

    def execute(self, task: str):
        """
        1) If no planner, do direct execution with the model (use background).
        2) If planner is GenericPlanner, run step-by-step (use background) and record each step.
        3) If planner is GraphPlanner, call .plan(task), which automatically runs node-based planning
           (using knowledge for plan generation and background for node execution).
        """
        self.logger.info(f"Agent is executing task: {task}")

        # Case 1: No planner => direct single-step
        if not self._planner:
            # Possibly build a context_section from the context
            context_section = self._context.context_to_str()
            final_prompt = self._execute_prompt.format(
                context_section=context_section,
                background=self.background,
                task=task,
            )
            response = self._model.process(final_prompt)
            self.logger.info(f"Response: {response}")
            self._execution_history.append(
                {
                    "step_name": "Direct Task Execution",
                    "step_description": task,
                    "step_result": str(response),
                }
            )
            return response

        # Case 2: Using a planner => pass knowledge to the plan
        steps = self._planner.plan(task, self.tools, knowledge=self.knowledge)

        # If the planner is GraphPlanner, .plan() already calls execute_plan() internally.
        # So we do NOT do the step-based for-loop here.
        if isinstance(self._planner, GraphPlanner):
            self._execution_history = steps
            # Return after the graph-based plan is done
            return "Task execution completed using GraphPlanner."

        # Otherwise, it's a GenericPlanner => do step-based execution
        if isinstance(self._planner, GenericPlanner):
            # (Note: If it's a GraphPlanner subclassing GenericPlanner, you'd hit the above if-check first.)
            self.logger.info(f"Executing plan with {len(steps)} steps.")
            for idx, step in enumerate(steps, 1):
                context_section = self._context.context_to_str()
                # Possibly incorporate the context in the step prompt
                final_prompt = self._execute_prompt.format(
                    context_section=context_section,
                    background=self.background,
                    task=step.description,
                )
                self.logger.info(f"Executing Step {idx}: {step.description}")
                response = self._model.process(final_prompt)
                self.logger.info(f"Response for Step {idx}: {response}")

                # Record the step execution
                self._execution_history.append(
                    {
                        "step_name": step.name,
                        "step_description": step.description,
                        "step_result": str(response),
                    }
                )
            return "Task execution completed using GenericPlanner."

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

        # Build a textual representation of the execution history
        history_lines = []
        for idx, record in enumerate(self._execution_history, 1):
            line = (
                f"Step {idx}: {record['step_name']}\n"
                f"Description: {record['step_description']}\n"
                f"Result: {record['step_result']}\n"
            )
            history_lines.append(line)

        history_text = "\n".join(history_lines)
        final_prompt = self._summary_prompt.format(history_text=history_text)

        self.logger.info("Generating final execution result (summary).")
        summary_response = self._model.process(final_prompt)
        return str(summary_response)
