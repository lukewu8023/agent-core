# planners/generic_planner.py

from typing import List, Optional, Dict, Any

from pydantic import BaseModel

from .base_planner import BasePlanner
from ..entities.agent_tool import AgentTool
from ..entities.steps import Steps, Step
from ..evaluators import BaseEvaluator
from ..evaluators.entities.evaluator_result import EvaluatorResult
from ..utils.context_manager import ContextManager

JSON_FORMAT = "```json"

NO_PLANNER = """
Given the following task and the tools, generate a step.

**Instructions for generating 'use_tool'**
If the **Tools** section is empty, set "use_tool" to false for all steps and omit "tool_name."
If the **Tools** section contains tools, set "use_tool" to true when a tool is necessary. Include "tool_name" in those steps and reference any tool-specific properties or arguments in the description.

**Task Breakdown Requirements**
1) The step must be encapsulated under the "steps" key in valid JSON format.
2) The step should include:
    "name": The name of the step
    "description": A concise description of the action to be performed in that step
    "use_tool": A boolean indicating whether a tool should be used
    Optionally, "tool_name": The name of the tool if "use_tool" is true
    "category": Categorize the step based on its function ({categories_str})
3) The possible categories for each step are: {categories_str}.
    If you cannot fit into any existing category, define a new category in "category".

**Background**
{background}

**Knowledge**
{knowledge}

**Tools**
{tools_knowledge}

**Task**
{task}

**Examples**
{example_json1}
{example_json2}

**Note:** 
Ensure your response is valid JSON, without any additional text or comments.

**Steps:**
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
        }
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
        }
    ]
}"""

DEFAULT_EXECUTE_PROMPT = r"""
Based on the below background, context and failed history, process the following current task, being mindful not to repeat or reintroduce errors from previous failed attempts, and respond with those suggestions:

**Background**
{background}

**Context**
{context}

**Failed History**
{failure_info}

**Current Task**
{task}

**Use Tool**
{use_tool}

**Tool Usage Guide**
Task Tool Description: {tool_description}
If Task Use Tool is `False`, process according to the current task,
If Task Use Tool is `True`, process using tools,
For each tool argument, based on context and human's question to generate arguments value according to the argument description.

**Output Example**
If task use tool is true, example:
```json
{{
    "use_tool": true,
    "tool_name": "Event",
    "tool_arguments": {{
        "eventId": "1000"
    }}
}}
```
If task use tool is false, example:
```json
{{
    "use_tool": false,
    "response": "<string>"
}}
```

**Note** 
1. The response must be a valid JSON object.
2. The response must be directly parsable by `json.loads()`, without syntax errors.
3. Do not include comments, explanations, or extra text outside the JSON block.
4. Ensure that all strings are properly enclosed in double quotes (").
5. Escape only necessary special characters:
   - Newlines as `\\n`
   - Double quotes inside strings as `\\"`
   - Backslashes as `\\`
6. Do NOT include extra escape sequences (e.g., avoid `\\\\` where `\\` is enough).
7. The response must not contain trailing commas or missing brackets.
8. Please make sure one step only use one tool one time, if need to use multiple tools or one tool multiple times, please separate them into different steps.
"""


class ExecuteResult(BaseModel):
    use_tool: bool
    response: Optional[Any] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[Any] = None


async def success_result(context_manager: ContextManager, execution_history: Steps, step: Step):
    context_manager.add_context(step.name, step.to_success_info())
    execution_history.add_success_step(step)


class NoPlanner(BasePlanner):
    """
    A simple planner that calls the model to break a task into JSON steps.
    Each step may optionally specify a category, used for specialized evaluation.
    """

    def __init__(self, model_name: str = None, log_level: Optional[str] = None):
        """
        If 'model' is not provided, the default model from config will be used.
        'prompt' can override the default prompt used for planning.
        """
        super().__init__(model_name, log_level)
        self.agent_tool = None

    async def plan(
        self,
        task: str,
        agent_tool: Optional[AgentTool],
        knowledge: str = "",
        background: str = "",
        categories: Optional[List[str]] = None,
    ) -> List[Step]:
        """
        Use the LLM to break down the task into multiple steps in JSON format.
        'knowledge' is appended to the prompt to guide the planning process.
        If 'categories' is provided, we pass it to the LLM so it can properly categorize each step.
        """
        self.logger.info(f"Creating plan for task: {task}")
        self.agent_tool = agent_tool

        categories_str = ", ".join(categories) if categories else "(Not defined)"

        final_prompt = NO_PLANNER.format(
            knowledge=knowledge,
            background=background,
            task=task,
            tools_knowledge=agent_tool.get_tool_knowledge(),
            example_json1=EXAMPLE_JSON1,
            example_json2=EXAMPLE_JSON2,
            categories_str=categories_str,
        )

        response_text = await self._model.process(final_prompt)

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
        self.logger.info(f"Plan: \n{plan}")
        return plan.steps

    async def execute_plan(
        self,
        plan: List[Step],
        task: str,
        execution_history: Steps,
        evaluators_enabled: bool,
        evaluators: dict,
        context_manager: ContextManager = ContextManager(),
        background: str = "",
    ):
        """
        Executes the PlanGraph node by node.
        'steps' is ignored in practice, because we use self.plan_graph.
        This signature is here for consistency with the BasePlanner interface.
        """
        self.logger.info(f"Executing plan with {len(plan)} steps.")
        step = plan[0]

        step, threshold = await self.execute(
            step, evaluators_enabled, task, background, evaluators, context_manager, None
        )
        current_attempts = 1
        if step.evaluator_result.score > threshold:
            await success_result(context_manager, execution_history, step)
        else:
            execution_history.add_retry_step(step)
            retry = True
            retry_steps: List[Step] = [step]
            while retry and 3 > current_attempts:
                attempt_step, threshold = await self.execute(
                    step,
                    evaluators_enabled,
                    task,
                    background,
                    evaluators,
                    context_manager,
                    retry_steps,
                )
                current_attempts = current_attempts + 1
                evaluator_result = attempt_step.evaluator_result
                if evaluator_result.score > threshold:
                    attempt_step.retries = retry_steps
                    step = attempt_step
                    retry = False
                else:
                    execution_history.add_retry_step(attempt_step)
                    retry_steps.append(attempt_step)
                    retry = True
            if not retry:
                await success_result(context_manager, execution_history, step)
            else:
                execution_history.add_failure_step(step)
        self.logger.info("Task execution completed using NoPlanner")

    async def execute(
        self,
        step,
        evaluators_enabled,
        task,
        background,
        evaluators,
        context_manager: ContextManager,
        failure_step: Optional[List[Step]],
    ) -> (Step, float):
        response = await self._execute_node(
            self.model_name, task, background, step, context_manager, failure_step
        )
        step.evaluator_result, threshold = await self._evaluate_node(
            step, task, response, evaluators_enabled, evaluators, context_manager, background
        )
        return step, threshold

    async def _execute_node(
            self,
            model_name: str,
            task: str,
            background: str,
            step: Step,
            context_manager: ContextManager,
            failure_step: List[Step],
    ) -> str:
        """
        Build prompt + call the LLM. If 'use_tool', invoke the tool.
        """
        failure_info = ""
        if failure_step:
            for f_step in failure_step:
                failure_info = (
                        failure_info
                        + f"Result : {f_step.result}, Result Suggestion: {f_step.evaluator_result.suggestion}\n"
                )

        tool_description = self.process_tool_description(step)

        # Node doesn't store a custom prompt, so we use self._execute_prompt
        final_prompt = DEFAULT_EXECUTE_PROMPT.format(
            context=context_manager.context_to_str(),
            task=task,
            background=background,
            use_tool=step.use_tool,
            tool_description=tool_description,
            failure_info=failure_info,
        )
        step.prompt = final_prompt

        # Use executor instead of direct model call
        response = await self.executor.execute(final_prompt, model_name)

        cleaned = response.replace(JSON_FORMAT, "").replace("```", "").strip()

        try:
            data = ExecuteResult.model_validate_json(cleaned)
            if data.use_tool:
                if step.tool_name:
                    try:
                        step.tool_args = data.tool_arguments
                        tool_response = await self.agent_tool.execute_tool(step.tool_name, data.tool_arguments)
                        response = f"""
tool description: {tool_description}
tool arguments: {data.tool_arguments} 
tool response : {tool_response}
"""
                    except Exception as e:
                        response = "Incorrect tool arguments and unexpected result when invoke the tool."
                else:
                    response = "Tool usage was requested, but no tool is attached to this node."
            else:
                response = data.response
        except Exception as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            response = f"Invalid JSON format in response : {cleaned}"
        self.logger.info(f"Response:\n {response}")
        step.result = response
        return response

    def process_tool_description(self, step: Step):
        tool_description = ""
        if step.use_tool:
            tool_description = self.agent_tool.get_tool_schema(step.tool_name)
        return tool_description

    async def _evaluate_node(
            self,
            step: Step,
            root_task: str,
            result: str,
            evaluators_enabled: bool,
            evaluators: Dict[str, BaseEvaluator],
            context_manager: ContextManager,
            background: str,
    ) -> (EvaluatorResult, float):
        """
        evaluate the node output using agent's evaluator if enabled.
        Return 0..1 scale.
        """
        if not evaluators_enabled:
            return EvaluatorResult(), 0.0
        chosen_cat = step.category if step.category in evaluators else "default"
        evaluator = evaluators.get(chosen_cat)
        if not evaluator:
            self.logger.warning(
                f"No evaluator found for category '{chosen_cat}'. evaluation skipped."
            )
            return EvaluatorResult(), 0.0
        evaluator_result = await evaluator.evaluate(
            root_task, step.description, result, background, context_manager
        )
        self.logger.info(
            f"Node {step.name} evaluation result: {evaluator_result.to_log()}"
        )
        return evaluator_result, evaluator.evaluation_threshold









