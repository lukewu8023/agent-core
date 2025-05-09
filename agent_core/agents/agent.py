# agents/agent.py

import json
import datetime
import os
import re
from typing import Optional, List

from langchain_core.tools import BaseTool
from agent_core.agent_basic import AgentBasic
from agent_core.entities.steps import Steps, Step, Summary
from agent_core.models.model_registry import ModelRegistry
from agent_core.planners.base_planner import BasePlanner
from agent_core.utils.context_manager import ContextManager
from agent_core.evaluators.evaluators import get_evaluator
from agent_core.evaluators import BaseEvaluator
from agent_core.utils.narrative_templates import (
    EXECUTION_NARRATIVE_TEMPLATES,
    PLAN_NARRATIVE_TEMPLATES,
)

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


class Agent(AgentBasic):
    """
    The Agent coordinates task execution with or without a Planner.
    It now exposes two prompts:
      - execute_prompt: overrides how we generate the no-planner prompt
      - summary_prompt (used in get_execution_result)
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
        self.execute_prompt = DEFAULT_EXECUTE_PROMPT
        self.summary_prompt = DEFAULT_SUMMARY_PROMPT
        self.response_prompt = DEFAULT_FINAL_RESPONSE_PROMPT

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
        self._execution_history.add_plan(plan)
        # Now just call planner's execute_plan(...) in a unified way
        self.planner.execute_plan(
            task=task,
            execution_history=self._execution_history,
            plan=plan,
            context_manager=self.context,
            background=self.background,
            evaluators_enabled=self.evaluators_enabled,
            evaluators=self.evaluators,
        )
        agent_result = self.get_execution_result_summary()
        self.get_token()
        return agent_result.output_result

    def execute_without_planner(self, task: str):
        context_section = self.context.context_to_str()
        final_prompt = self.execute_prompt.format(
            context_section=context_section,
            background=self.background,
            task=task,
        )
        response = self._model.process(final_prompt)
        self.logger.info(f"Response: {response}")
        self._execution_history.add_success_step(
            Step(
                name="Direct Task Execution",
                description=task,
                result=str(response),
                prompt=final_prompt,
            )
        )
        return response

    def planner(self, planner):
        if not issubclass(planner.__class__, BasePlanner):
            error_msg = "Planner must be an instance of BasePlanner."
            self.logger.error(error_msg)
            raise TypeError(error_msg)
        self.planner = planner
        self.logger.info(f"Agent planner uses model: {planner.__class__.__name__}")

    def enable_evaluators(self):
        self.evaluators_enabled = True
        self.logger.info("Evaluators have been enabled.")

    def disable_evaluators(self):
        self.evaluators_enabled = False
        self.logger.info("Evaluators have been disabled.")

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

    @property
    def execution_history(self) -> Steps:
        """
        Read-only access to the execution history.
        Each item is a dict with keys: 'step_name', 'step_description', 'step_result'.
        """
        return self._execution_history

    def get_execution_history(self):
        self._execution_history.execution_history_to_str()

    def export_execution_trace(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_name = f"logs/trace_{timestamp}.json"
        os.makedirs(os.path.dirname(trace_name), exist_ok=True)
        with open(trace_name, "w", encoding="utf-8") as f:
            json.dump(
                self.execution_history.model_dump(), f, indent=4, ensure_ascii=False
            )

    @property
    def execution_responses(self) -> str:
        """
        Read-only access to the execution responses.
        Combine all 'step_result' together.
        """
        return self._execution_history.execution_history_to_responses()

    def get_final_response(self, task: str) -> str:
        history_text = self._execution_history.execution_history_to_str()
        final_response_prompt = self.response_prompt.format(
            task=task, history_text=history_text
        )
        self.logger.info("Generating final response.")
        final_response = self._model.process(final_response_prompt)
        return str(final_response)

    def get_execution_result_summary(self) -> Summary:
        """
        Produce an overall summary describing how the solution was completed,
        using the LLM (agent's model) to format the final explanation if desired.
        """
        if not self._execution_history:
            return Summary(
                summary="", output_result=("No direct step-based execution history recorded. "
                "(If you used GraphPlanner, the node-based execution is stored inside the planner.)"
            ), conclusion="")

        history_text = self._execution_history.execution_history_to_str()
        final_prompt = self.summary_prompt.format(history_text=history_text)

        self.logger.info("Generating execution result summary.")
        summary_response = self._model.process(final_prompt)
        cleaned = summary_response.replace("```json", "").replace("```", "").strip()
        summary = Summary.model_validate_json(cleaned)
        self._execution_history.summary = summary
        return summary

    def get_execution_reasoning(self):
        """
        Generate a narrative describing the agent's reasoning process based on the execution history.
        Includes both the plans and the execution steps in a coherent narrative.

        Returns:
            str: A narrative description of the reasoning process
        """
        narrative_parts = []

        # Check if we have plans and steps to process
        if (
            not hasattr(self._execution_history, "trace_plan")
            or not self._execution_history.trace_plan
        ):
            return []

        # Get the initial plan (plan_id = 1)
        initial_plan = self._execution_history.trace_plan.get(1)
        if initial_plan and initial_plan.plan:
            plan_steps = "\n".join(
                [f"*Step {step.name}: {step.description}" for step in initial_plan.plan]
            )
            narrative_parts.append(
                self._get_plan_narrative(
                    template_key="initial_plan", plan_steps=plan_steps
                )
            )

        # Get all steps in the trace
        steps = (
            self._execution_history.trace_steps
            if hasattr(self._execution_history, "trace_steps")
            else []
        )

        # Process each step and add new plans when they appear
        current_plan_id = 1
        for step in steps:

            # First add the reasoning for the current step
            narrative_parts.append(self._get_step_narrative(step))
            next_plan_id = current_plan_id + 1

            # Check if this step triggered a plan change
            if (not step.action.startswith("failure ") and not step.action.startswith("success "))\
                    or next_plan_id not in self._execution_history.trace_plan:
                continue

            # Then describe the new plan using the appropriate template
            next_plan = self._execution_history.trace_plan[next_plan_id]
            if next_plan and next_plan.plan:
                self.process_step(next_plan, narrative_parts, step)
                current_plan_id = next_plan_id

        return narrative_parts

    def process_step(self, next_plan, narrative_parts, step):
        plan_steps = "\n".join(
            [
                f"*Step {step.name}: {step.description}"
                for step in next_plan.plan
            ]
        )
        modifications = None
        new_tasks = None
        if next_plan.adjustment:
            modifications = "\n".join(
                [
                    f"*Step {adjust.name}: {adjust.description}"
                    for adjust in next_plan.adjustment.modifications
                ]
            )
            new_tasks = "\n".join(
                [
                    f"*Step {adjust.name}: {adjust.description}"
                    for adjust in next_plan.adjustment.new_subtasks
                ]
            )

        # Get plan narrative based on the step
        narrative_parts.append(
            self._get_plan_narrative(step=step, plan_steps=plan_steps,
                                     modifications=modifications, new_tasks=new_tasks)
        )

    def _get_plan_narrative(self, plan_steps, modifications=None, new_tasks=None,
                            step=None, template_key=None):
        """
        Get a formatted plan narrative using the appropriate template.

        Args:
            plan_steps: String containing the steps description
            step: Step object that triggered the plan change (optional)
            template_key: Override template key (used for initial plan)

        Returns:
            str: Formatted plan narrative
        """
        # If template_key is not provided, determine it from the step
        if template_key is None and step is not None:
            if "failure" in step.action:
                template_key = (
                    "breakdown_plan" if "breakdown" in step.action else "failure_replan"
                )
            elif "success" in step.action:
                template_key = "success_replan"
            else:
                template_key = "replan"
        elif template_key is None:
            template_key = "replan"  # Default fallback

        template = PLAN_NARRATIVE_TEMPLATES.get(
            template_key, PLAN_NARRATIVE_TEMPLATES["replan"]
        )
        return template.format(plan_steps=plan_steps, modifications=modifications, new_tasks=new_tasks)

    def _get_step_narrative(self, step):
        """
        Generate a narrative string for a given step based on its action.

        Args:
            step: A Step object containing action and other metadata

        Returns:
            str: A narrative description of the reasoning at this step
        """
        action = step.action
        # Handle special cases with combined actions
        if action.startswith("failure "):
            if "breakdown" in action:
                template_key = "failure breakdown"
            elif "replan" in action:
                template_key = "failure replan"
            else:
                template_key = "failure"
        elif action.startswith("success "):
            if "replan" in action:
                template_key = "success replan"
            elif "none" in action:
                template_key = "success none"
            else:
                template_key = "success replan"  # default to replan
        else:
            template_key = action

        # Use the appropriate template or a default one
        template = EXECUTION_NARRATIVE_TEMPLATES.get(
            template_key, "Taking step {step_name}."
        )

        result = step.result
        if step.use_tool:
            pattern = r"tool description: (.*?)tool arguments: (.*?)tool response : (.*)"
            match = re.search(pattern, step.result, re.DOTALL)
            if match:
                result = match.group(3).strip()
        # Format the template with step data
        return template.format(step_name=step.name, step_description=step.description, step_result=result,
                               step_suggestion=step.evaluator_result.suggestion)

    def get_token(self):
        input_tokens, output_tokens = ModelRegistry.get_token()
        self._execution_history.input_tokens = input_tokens
        self._execution_history.output_tokens = output_tokens
