# planners/generic_planner.py

from typing import List, Optional, Dict
from langchain_core.tools import BaseTool
from .base_planner import BasePlanner, tool_knowledge_format
from ..entities.steps import Steps, Step
from ..evaluators import BaseEvaluator
from ..utils.context_manager import ContextManager


class GenericPlanner(BasePlanner):
    """
    A simple planner that calls the model to break a task into JSON steps.
    Each step may optionally specify a category, used for specialized evaluation.
    """

    EXAMPLE_JSON1 = """{
        "steps": [
            {
                "name": "Prepare eggs",
                "description": "Get the eggs from the fridge and put them on the table.",
                "use_tool": true,
                "tool_name": "Event",
                "category": "action",
                "evaluation_threshold": 0.9 // define threshold for evaluation process, 0.0 to 1.0, more complex task more lower threshold
            },
            ...
        ]
    }"""

    EXAMPLE_JSON2 = """{
        "steps": [
            {
                "name": "Plan code structure",
                "description": "Outline the classes and methods.",
                "use_tool": false,
                "category": "coding",
                "evaluation_threshold": 0.9 // define threshold for evaluation process, 0.0 to 1.0, more complex task more lower threshold
            },
            ...
        ]
    }"""

    def __init__(self, model_name: str = None, log_level: Optional[str] = None):
        """
        If 'model' is not provided, the default model from config will be used.
        'prompt' can override the default prompt used for planning.
        """
        super().__init__(model_name, log_level)

    def plan(
        self,
        task: str,
        tools: Optional[List[BaseTool]],
        knowledge: str = "",
        background: str = "",
        categories: Optional[List[str]] = None,
    ) -> Steps:
        """
        Use the LLM to break down the task into multiple steps in JSON format.
        'knowledge' is appended to the prompt to guide the planning process.
        If 'categories' is provided, we pass it to the LLM so it can properly categorize each step.
        """
        self.logger.info(f"Creating plan for task: {task}")

        tools_knowledge = tool_knowledge_format(tools)
        categories_str = ", ".join(categories) if categories else "(Not defined)"

        final_prompt = self.prompt.format(
            knowledge=knowledge,
            background=background,
            task=task,
            tools_knowledge=tools_knowledge,
            example_json1=self.EXAMPLE_JSON1,
            example_json2=self.EXAMPLE_JSON2,
            categories_str=categories_str,
        )

        response_text = self._model.process(final_prompt)

        if not response_text or not response_text.strip():
            error_msg = "LLM returned an empty or null response."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.debug(f"Raw LLM response: {repr(response_text)}")

        # Minor cleanup of possible code fences
        cleaned = response_text.replace("```json", "").replace("```", "").strip()

        try:
            plan = Steps.model_validate_json(cleaned)
        except Exception as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.error(f"Raw LLM response was: {cleaned}")
            raise ValueError("Invalid JSON format in planner response.")
        self.logger.info(f"Got {len(plan.steps)} steps from the LLM.")
        return plan

    def execute_plan(
        self,
        plan: Steps,
        task: str,
        evaluators_enabled: bool,
        evaluators: Dict[str, BaseEvaluator],
        context_manager: ContextManager,
        background: str = "",
    ):
        """
        Execute a list of steps (previously planned).
        This replaces the step-by-step logic that was inside agent.py for GenericPlanner.
        """
        self.logger.info(f"Executing plan with {len(plan.steps)} steps.")

        for idx, step in enumerate(plan.steps, 1):
            context_section = (
                context_manager.context_to_str() if context_manager else ""
            )
            final_prompt = f"""
                {context_section}\n
                **Background**
                {background}\n
                ***Root Task***
                {task}\n
                **Sub Task**
                {step.description}
            """
            step.prompt = final_prompt
            self.logger.info(f"Executing Step {idx}: {step.description}")
            response = self._model.process(final_prompt)
            step.result = response
            self.logger.info(f"Response for Step {idx}: {response}")

            # Optional Evaluation
            if evaluators_enabled:
                attempt = 1
                chosen_cat = step.category if step.category in evaluators else "default"
                evaluator = evaluators.get(chosen_cat)
                evaluator_result = evaluator.evaluate(
                    task, step.description, response, background, context_manager
                )
                step.add_evaluator_result(evaluator_result)
                self.logger.info(evaluator_result.to_log())

                while (
                    evaluator_result.score <= evaluator.evaluation_threshold
                    and evaluator.max_attempt + 1 > attempt
                ):
                    self.logger.info(
                        f"Executing Step {idx} Failed Attempt {attempt}: {step.description}"
                    )
                    retry_step = Step(name=step.name, description=step.description)
                    retry_prompt = f"""
                        {final_prompt}\n
                        **Failed Evaluate Response**
                        {response} 
                        **Evaluator**
                        {evaluator_result.details}
                        """
                    retry_step.prompt = retry_prompt
                    response = self._model.process(retry_prompt)
                    retry_step.result = response
                    evaluator_result = evaluator.evaluate(
                        task, step.description, response, background, context_manager
                    )
                    retry_step.add_evaluator_result(evaluator_result)
                    step.add_retry(retry_step)
                    self.logger.info(
                        f"Response for Rerun Step {idx} Failed Attempt {attempt}: {response}"
                    )
                    attempt = attempt + 1

            if context_manager:
                context_manager.add_context("Execution History", step.to_success_info())
            else:
                (
                    context_manager.get_context_by_key("Execution History")
                    + step.to_success_info()
                )
        return plan
