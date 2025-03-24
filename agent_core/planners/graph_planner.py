# planners/graph_planner.py

from agent_core.evaluators import BaseEvaluator
from agent_core.evaluators.entities.evaluator_result import EvaluatorResult
from pydantic import BaseModel
from agent_core.planners.base_planner import (
    BasePlanner,
    tool_knowledge_format,
)
from agent_core.planners.generic_planner import GenericPlanner
from agent_core.models.model_registry import ModelRegistry
from agent_core.utils.context_manager import ContextManager
from agent_core.entities.steps import Steps, Step
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from langchain_core.tools import BaseTool
from agent_core.utils.logger import get_logger

DEFAULT_EXECUTE_PROMPT = r"""
Based on the below background, context and failed history, process the following current task, being mindful not to repeat or reintroduce errors from previous failed attempts, and respond with those suggestions:

**Background**
{background}

**Context**
{context}

**Failed History**
{failure_info}

**Root Task**
{task}

**Current Task**
{description}

**Use Tool**
{use_tool}

**Tool Usage Guide**
Task Tool Description: {tool_description}
If Task Use Tool is `False`, process according to the description of the current task,
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
"""

DEFAULT_FAILURE_REPLAN_PROMPT = """
You are an intelligent assistant helping to adjust a task execution plan represented as a graph of subtasks. 
Below are the details:

**Background**
{background}

**Knowledge**
{knowledge}

**Tools**
{tools_knowledge}

**Categories**
{categories_str}

**Root Task**
{root_task}

**Current Plan**
{execution_plan}

**Execution History**
{execution_history}

**Current Node Name**
{current_node_name}

**Current Node Execution Failure Reason**
{failure_reason}

**Replanning History**
{replan_history}

**Instructions**
- Analyze the Current Plan, Execution History, Failure Reason and Replanning History to decide on one of two actions:
    1. **breakdown**: Break down the task of failed node {current_node_name} into smaller subtasks.
    2. **replan**: Go back to a previous node for replanning, 
- If you choose **breakdown**, provide detailed descriptions of the new subtasks, only breakdown the current (failed) node, otherwise it should be replan. ex: if current node is B, breakdown nodes should be B.1, B.2, if current node is B.2, breakdown nodes should be B.2.1, B.2.2... and make the all nodes as chain eventually.
- If you choose **replan**, specify which node to return to and suggest any modifications to the plan after that node, do not repeat previous failure replanning in the Replanning History.
- The name generated following the naming convention as A.1, B.1.2, C.2.5.2, new name (not next_nodes) generation example: current: B > new sub: B.1, current: B.2.2.2 > new sub: B.2.2.2.1
- Return your response in the following JSON format (do not include any additional text):

```json
{{
    "action": "breakdown" or "replan",
    "new_subtasks": [  // Required if action is "breakdown"
        {{
            "name": "...", // unique task name
            "description": "...", // description of the subtask
            "next_node": "...", // next node name
            "evaluation_threshold": 0.8, // it can be changed based on the complexity of the task
            "max_attempts": 3
        }}
    ],
    "restart_node_name": "...", // Required if action is "replan"
    "modifications": [
        {{
            "name": "...",
            "description": "...",
            "next_node": "...",
            "evaluation_threshold": 0.8, // it can be changed based on the complexity of the task
            "max_attempts": 3
        }}
    ],
    "rationale": "..." // explanation of your reasoning here
}}
```

**Note** 
Ensure your response is valid JSON, without any additional text or comments (// explain).
"""

# post-success replan prompt:
DEFAULT_SUCCESS_REPLAN_PROMPT = """
You are an intelligent planner reviewing a plan after the successful execution of a subtask. Your goal is to assess whether the current plan still effectively achieves the root task, or if minimal adjustments to future unexecuted steps are necessary (maybe some information is missing or an additional tool call is required to get the required information). Only replan if absolutely required. Prefer leaving the plan unchanged if it remains sufficient.

**Background**
{background}

**Knowledge**
{knowledge}

**Tools**
{tools_knowledge}

**Categories**
{categories_str}

**Root Task**
{root_task}

**Current Node Name**
{current_node_name}

**Executed Plan (Each node with execution results, if executed)**
{executed_plan}

**Remaining Plan**
{remaining_plan}

**Instructions**
Decide between:
1. Do nothing (action = "none") if the remaining plan is likely to fulfill the root task and no critical changes are needed.
2. Replan (action = "replan") only if there's a clear gap or something essential missing in the upcoming steps (for example, need to retrieve required information from a tool, then add a step of the tool calling)

When replanning:
- Modify only the future (unexecuted) steps.
- Name new nodes following the hierarchy (e.g., current `B` → new sub `B.1`).
- Keep additions minimal and targeted; aim for simplicity.

**Example**
```json
{{
    "action": "none" or "replan",
    "modifications": [ // build full steps to replace the un-executed steps
        {{
            "name": "...",
            "description": "...",
            "next_node": "...", // leave it empty string if it is the last node
            "evaluation_threshold": 0.8, // it can be changed based on the complexity of the task
            "max_attempts": 3
        }}
    ],
    "rationale": "..." // explanation of your reasoning here
}}
```
(If “action” = “none”, leave “modifications” as empty arrays.)

**Data Example**
if no change on the plan:
```json
{{
    "action": "none",
    "modifications": [],
    "rationale": "The remaining steps are adequate to achieve the root task. No replanning required."
}}
```
or, if minimal replanning is necessary:
```json
{{
    "action": "replan",
    "modifications": [ // build full steps to replace the un-executed steps
        {{
            "name": "B.2.1",
            "description": "Obtain missing data required to proceed.",
            "next_node": "B.3", 
            "evaluation_threshold": 0.8,
            "max_attempts": 3
        }}
    ],
    "rationale": "The original plan lacked critical information to proceed, hence a minimal adjustment is required."
}}
```

**Important**
- Ensure your response is valid JSON without any additional text or comments (// explain).
- Default action should strongly favor “none”. Only replan if absolutely essential.
"""


class Node(Step):
    """
    Represents a single node in the PlanGraph.
    """

    next_node: Optional[str] = ""
    tool: Optional[BaseTool] = None
    evaluation_threshold: Optional[float] = None
    max_attempts: int = 3
    current_attempts: int = 0

    def set_next_node(self, node: "Node"):
        if not self.next_node or self.next_node == "":
            self.next_node = node.name


@dataclass
class PlanGraph:
    """
    Holds multiple Node objects in a directed structure.
    Node-based plan execution with possible replan logic.
    """

    logger = get_logger("plan-graph")

    nodes: Dict[str, Node] = field(default_factory=dict)
    start_node_name: Optional[str] = None
    replan_history: List[Dict] = field(default_factory=list)
    current_node_name: Optional[str] = None

    # We'll store some reference info for replan
    background: str = ""
    knowledge: str = ""
    categories: Optional[List[str]] = None
    task: str = ""
    tools: str = ""  # textual representation from tool_knowledge_format
    prompt: str = ""  # The replan prompt if needed

    def add_node(self, node: Node):
        self.nodes[node.name] = node
        if self.start_node_name is None:
            self.start_node_name = node.name

    def to_plan(self) -> List[Step]:
        plan = list()
        process_node_name = self.start_node_name
        while process_node_name and process_node_name != "":
            process_node = self.nodes.get(process_node_name)
            plan.append(process_node)
            process_node_name = process_node.next_node
        return plan

    def add_history_record(self, record: Dict):
        self.replan_history.append(record)

    def summarize_plan(self) -> str:
        summary = ""
        for n in self.nodes.values():
            summary += f"Node {n.name}: {n.description}, "
            if n.result:
                summary += f"Node {n.name} Result {n.result}, "
            summary += f"Next Node: {n.next_node}\n"
        return summary

    def execution_plan(self) -> str:
        execution_plan = ""
        for n in self.nodes.values():
            execution_plan += f"Node {n.name}: {n.description}, "
            execution_plan += f"Next Node: {n.next_node}\n"
        return execution_plan

    def executed_plan(self) -> str:
        executed_plan = ""
        for n in self.nodes.values():
            executed_plan += f"Node {n.name}: {n.description}, "
            if n.result:
                executed_plan += f"Node {n.name} Result {n.result}, "
                executed_plan += f"Next Node: {n.next_node}\n"
            else:
                break
        return executed_plan

    def remaining_plan(self) -> str:
        remaining_plan = ""
        for n in self.nodes.values():
            if n.result:
                continue
            else:
                remaining_plan += f"Node {n.name}: {n.description}, "
                remaining_plan += f"Next Node: {n.next_node}\n"
        return remaining_plan


@dataclass
class Adjustments(BaseModel):
    action: str
    rationale: str
    new_subtasks: Optional[List[Node]] = field(
        default_factory=dict
    )  # Provide an empty list as default
    restart_node_name: Optional[str] = None  # Allow None if missing
    modifications: Optional[List[Node]] = field(
        default_factory=dict
    )  # Provide an empty list as default


class ExecuteResult(BaseModel):
    use_tool: bool
    response: Optional[Any] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[Any] = None


def cleanup_context(execution_history: Steps, restart_node_name):
    remove_index = -1
    for index, step in enumerate(execution_history.steps):
        if step.name == restart_node_name:
            remove_index = index
            break
    if remove_index != -1:
        execution_history.steps = execution_history.steps[:remove_index]


def pass_threshold(score, node_threshold, evaluator_threshold):
    if node_threshold:
        return score >= node_threshold
    else:
        return score >= evaluator_threshold


class GraphPlanner(BasePlanner):
    """
    A planner that builds a PlanGraph, uses context, and executes node-based logic with re-planning.
    """

    def __init__(self, model_name: str = None, log_level: Optional[str] = None):
        super().__init__(model_name, log_level)
        self.plan_graph: Optional[PlanGraph] = None
        self.context_manager = ContextManager()

        self._execute_prompt = DEFAULT_EXECUTE_PROMPT
        self._failure_replan_prompt = DEFAULT_FAILURE_REPLAN_PROMPT
        self._success_replan_prompt = DEFAULT_SUCCESS_REPLAN_PROMPT

    @property
    def failure_replan_prompt(self) -> str:
        return self._failure_replan_prompt

    @failure_replan_prompt.setter
    def failure_replan_prompt(self, value: str):
        self._failure_replan_prompt = value

    @property
    def success_replan_prompt(self) -> str:
        return self._success_replan_prompt

    @success_replan_prompt.setter
    def success_replan_prompt(self, value: str):
        self._success_replan_prompt = value

    @property
    def execute_prompt(self) -> str:
        return self._execute_prompt

    @execute_prompt.setter
    def execute_prompt(self, value: str):
        self._execute_prompt = value

    def plan(
        self,
        task: str,
        tools: Optional[List[BaseTool]],
        knowledge: str = "",
        background: str = "",
        categories: Optional[List[str]] = None,
    ) -> Steps:
        """
        1) Call GenericPlanner to obtain a list of Steps using the same arguments.
        2) Convert those Steps into a PlanGraph with Node objects.
        3) Return the same Steps (for reference), but we'll actually execute nodes later.
        """
        self.logger.info(f"GraphPlanner is creating plan for task: {task}")

        # Use GenericPlanner internally to get the steps
        generic_planner = GenericPlanner(model_name=self.model_name, log_level=None)
        generic_planner.prompt = self.prompt
        plan = generic_planner.plan(
            task=task,
            tools=tools,
            knowledge=knowledge,
            background=background,
            categories=categories,
        )

        # Convert Steps -> Node objects in a new PlanGraph
        plan_graph = PlanGraph()
        plan_graph.prompt = self._failure_replan_prompt  # If needed for replan calls
        plan_graph.background = background
        plan_graph.knowledge = knowledge
        plan_graph.categories = categories
        plan_graph.task = task
        plan_graph.tools = tool_knowledge_format(tools)

        tool = None
        previous_node = None
        tool_map = {}
        if tools is not None:
            tool_map = {tool.name: tool for tool in tools}

        for idx, step in enumerate(plan.steps, start=1):
            node_name = chr(65 + idx - 1)  # e.g., A, B, C...
            next_node_name = chr(65 + idx) if idx < len(plan.steps) else ""

            if step.tool_name and tool_map:
                tool = tool_map.get(step.tool_name)
            node = Node(
                name=node_name,
                description=step.description,
                use_tool=step.use_tool,
                tool_name=step.tool_name,
                tool=tool,
                next_node=next_node_name,
                category=step.category,
            )
            plan_graph.add_node(node)

            if previous_node:
                previous_node.set_next_node(node)
            previous_node = node

        self.plan_graph = plan_graph
        return plan

    def execute_plan(
        self,
        plan: Steps,
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
        if not self.plan_graph:
            self.logger.error(
                "No plan graph found. Need to generate plan graph by plan() first."
            )
            return execution_history
        if context_manager:
            self.context_manager = context_manager
        pg = self.plan_graph
        pg.current_node_name = pg.current_node_name or pg.start_node_name
        while pg.current_node_name:
            if pg.current_node_name not in pg.nodes:
                self.logger.error(
                    f"Node {pg.current_node_name} does not exist in the plan. Aborting execution."
                )
                break
            node = pg.nodes[pg.current_node_name]
            step, threshold = self.execute(
                node, evaluators_enabled, task, background, evaluators, None
            )
            if pass_threshold(
                step.evaluator_result.score, node.evaluation_threshold, threshold
            ):
                if node.evaluation_threshold:
                    self.logger.info(
                        f"Actual Node Threshold: {node.evaluation_threshold}, Decision: Accept Output"
                    )
                self.success_result(node, execution_history, step)
                continue
            else:
                execution_history.add_retry_step(step)
                retry = True
                retry_steps: List[Step] = [step]
                while retry and node.max_attempts > node.current_attempts:
                    attempt_step, threshold = self.execute(
                        node,
                        evaluators_enabled,
                        task,
                        background,
                        evaluators,
                        retry_steps,
                    )
                    evaluator_result = attempt_step.evaluator_result
                    if pass_threshold(
                        evaluator_result.score, node.evaluation_threshold, threshold
                    ):
                        if node.evaluation_threshold:
                            self.logger.info(
                                f"Actual Node Threshold: {node.evaluation_threshold}, Decision: Accept Output"
                            )
                        attempt_step.retries = retry_steps
                        step = attempt_step
                        retry = False
                    else:
                        execution_history.add_retry_step(attempt_step)
                        retry_steps.append(attempt_step)
                        retry = True
                if not retry:
                    self.success_result(node, execution_history, step)
                    continue
                else:
                    self.logger.warning(f"Replanning is needed at Node {node.name}")
                    failure_info = self._prepare_failure_info(
                        execution_history, retry_steps[-1]
                    )
                    replan_response = self._failure_replan(pg, failure_info)
                    cleaned = (
                        replan_response.replace("```json", "")
                        .replace("```", "")
                        .strip()
                    )

                    try:
                        adjustments = Adjustments.model_validate_json(cleaned)
                    except Exception as e:
                        self.logger.error(f"Failed to parse JSON: {e}")
                        adjustments = None

                    if adjustments:
                        pg.add_history_record(
                            {
                                "name": node.name,
                                "replan_history": adjustments,
                            }
                        )
                        self.apply_adjustments_to_plan(node.name, adjustments, execution_history)
                        self.logger.info(
                            f"New plan after adjusted: {self.plan_graph.nodes}"
                        )
                        restart_node_name = self._determine_restart_node(adjustments)
                        if restart_node_name:
                            cleanup_context(
                                execution_history, restart_node_name
                            )
                            pg.current_node_name = restart_node_name
                        else:
                            break
                    else:
                        self.logger.error(
                            "Could not parse LLM response for replan, aborting."
                        )
                        break
        self.logger.info("Task execution completed using GraphPlanner")
        return execution_history

    def execute(
        self,
        node,
        evaluators_enabled,
        task,
        background,
        evaluators,
        failure_step: Optional[List[Step]],
    ) -> (Step, float):
        step = Step(
            name=node.name,
            description=node.description,
            use_tool=node.use_tool,
            tool_name=node.tool_name,
            category=node.category,
        )
        response = self._execute_node(
            node, self.model_name, task, background, step, failure_step
        )
        step.evaluator_result, threshold = self._evaluate_node(
            node, task, response, evaluators_enabled, evaluators, background
        )
        return step, threshold

    def success_result(self, node, execution_history: Steps, step: Step):
        # Add node info to context
        self.context_manager.add_context(step.name, step.to_success_info())
        # Keep the raw response in node's execution_results for reference
        # node.execution_results.append(step.result)
        execution_history.add_success_step(step)
        node.result = step.result
        # Post-success replan check
        # We'll see if we want to add or replace future steps on the fly.
        self._success_replan(self.plan_graph, node, execution_history)
        self.plan_graph.current_node_name = node.next_node

    def _execute_node(
        self,
        node: Node,
        model_name: str,
        task: str,
        background: str,
        step: Step,
        failure_step: List[Step],
    ) -> str:
        """
        Build prompt + call the LLM. If 'use_tool', invoke the tool.
        """
        self.logger.info(
            f"Executing Node {node.name} Attempt {node.current_attempts + 1}: {node.description}"
        )
        failure_info = ""
        if failure_step:
            for f_step in failure_step:
                failure_info = (
                    failure_info
                    + f"Result : {f_step.result}, Result Suggestion: {f_step.evaluator_result.suggestion}\n"
                )
        node.current_attempts += 1

        tool_description = ""
        if node.use_tool:
            if node.tool is None:
                self.logger.warning(
                    f"Node {node.name} indicates 'use_tool' but 'tool' is None. Skipping tool usage details."
                )
            elif hasattr(node.tool, "args_schema"):
                tool_description = str(node.tool.args_schema.model_json_schema())
            else:
                self.logger.warning(
                    f"Node {node.name} indicates 'use_tool' but the provided tool lacks 'args_schema'."
                )
                tool_description = f"[Tool: {node.tool_name}]"

        # Node doesn't store a custom prompt, so we use self._execute_prompt
        final_prompt = self._execute_prompt.format(
            context=self.context_manager.context_to_str(),
            task=task,
            background=background,
            description=f"Step {node.name}\nTask Desc: {node.description}",
            use_tool=node.use_tool,
            tool_description=tool_description,
            failure_info=failure_info,
        )
        step.prompt = final_prompt
        response = ModelRegistry.get_model(model_name).process(final_prompt)
        cleaned = response.replace("```json", "").replace("```", "").strip()

        try:
            data = ExecuteResult.model_validate_json(cleaned)
            if data.use_tool:
                if node.tool is not None:
                    try:
                        step.tool_args = data.tool_arguments
                        tool_response = node.tool.invoke(data.tool_arguments)
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

    def _evaluate_node(
        self,
        node: Node,
        root_task: str,
        result: str,
        evaluators_enabled: bool,
        evaluators: Dict[str, BaseEvaluator],
        background: str,
    ) -> (EvaluatorResult, float):
        """
        evaluate the node output using agent's evaluator if enabled.
        Return 0..1 scale.
        """
        if not evaluators_enabled:
            return EvaluatorResult(), 0.0
        chosen_cat = node.category if node.category in evaluators else "default"
        evaluator = evaluators.get(chosen_cat)
        if not evaluator:
            self.logger.warning(
                f"No evaluator found for category '{chosen_cat}'. evaluation skipped."
            )
            return EvaluatorResult(), 0.0
        evaluator_result = evaluator.evaluate(
            root_task, node.description, result, background, self.context_manager
        )
        self.logger.info(
            f"Node {node.name} evaluation result: {evaluator_result.to_log()}"
        )
        return evaluator_result, evaluator.evaluation_threshold

    def _prepare_failure_info(
        self, execute_history: Steps, current_failed: Step
    ) -> Dict:
        """
        Produce the context for replan prompt.
        """
        pg = self.plan_graph
        return {
            "failure_reason": (
                current_failed.evaluator_result.suggestion if current_failed else ""
            ),
            "execution_history": execute_history.get_info(),
            "replan_history": pg.replan_history,
        }

    def _success_replan(self, plan_graph: PlanGraph, node: Node, execution_history: Steps):
        """
        After a node is successfully executed, optionally adjust remaining steps
        in the plan using the DEFAULT_SUCCESS_REPLAN_PROMPT. If 'action' = 'none',
        do nothing; if 'action' = 'replan', we replace future steps with the
        modifications from the LLM.
        """
        # Prepare placeholders for the success replan prompt
        categories_str = (
            ", ".join(plan_graph.categories)
            if plan_graph.categories
            else "(Not defined)"
        )
        executed_plan = plan_graph.executed_plan()
        remaining_plan = plan_graph.remaining_plan()

        # Build the final prompt
        final_prompt = self._success_replan_prompt.format(
            background=plan_graph.background,
            knowledge=plan_graph.knowledge,
            tools_knowledge=plan_graph.tools,
            categories_str=categories_str,
            root_task=plan_graph.task,
            executed_plan=executed_plan,
            remaining_plan=remaining_plan,
            current_node_name=node.name,
        )

        self.logger.info("Calling model for success replan instructions...")
        response = self._model.process(final_prompt)
        self.logger.debug(f"Success replan response:\n{response}")
        cleaned = response.replace("```json", "").replace("```", "").strip()
        try:
            adjustments = Adjustments.model_validate_json(cleaned)
            if adjustments.action == "none":
                self.logger.info("No changes to the plan after success.")
                return
            elif adjustments.action == "replan":
                self.logger.info("Replanning the un-executed steps after success...")
                self.logger.info(f"New Steps: {cleaned}")
                # 1) Remove all un-executed steps from this node onward.
                #    This example removes any nodes listed in node.next_nodes, plus any "chain" from them.
                to_remove = [node.next_node]  # immediate next(s)
                # Optionally, you might do a BFS/DFS to remove all reachable children, if desired:
                # gather all reachable nodes in future
                stack = [node.next_node]
                visited = set()
                while stack:
                    nxt = stack.pop()
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    if nxt in plan_graph.nodes:
                        # add all *that* node’s next_nodes to stack
                        stack.extend(plan_graph.nodes[nxt].next_node)
                        to_remove.append(nxt)
                # Actually remove them from the plan
                for rm_name in to_remove:
                    if rm_name in plan_graph.nodes:
                        del plan_graph.nodes[rm_name]
                # 2) Add each modification as a new node (fully replacing the steps).
                for mod in adjustments.modifications:
                    if not mod.name:
                        self.logger.warning(
                            f"Skipping modification with no name: {mod}"
                        )
                        continue
                    plan_graph.add_node(mod)
                # 3) Link the current node to the first new node if it exists
                if adjustments.modifications:
                    node.next_node = adjustments.modifications[0].name
                else:
                    # If no new steps were added, then the plan ends here
                    node.next_node = ""
                self.logger.info(
                    "Successfully applied 'replan' modifications after success."
                )
            else:
                self.logger.warning(
                    f"Unknown action '{adjustments.action}' in success replan. No changes made."
                )
            execution_history.adjust_plan(adjustments.action, plan_graph.to_plan())
        except Exception as e:
            self.logger.warning(
                "Post-success replan response is not valid JSON. Skipping replan."
            )

    def _failure_replan(self, plan_graph: PlanGraph, failure_info: Dict) -> str:
        execution_plan = plan_graph.execution_plan()

        final_prompt = plan_graph.prompt.format(
            background=plan_graph.background,
            knowledge=plan_graph.knowledge,
            tools_knowledge=plan_graph.tools,
            root_task=plan_graph.task,
            context_str=self.context_manager.context_to_str(),
            categories_str=(
                ", ".join(plan_graph.categories)
                if plan_graph.categories
                else "(Not defined)"
            ),
            execution_plan=execution_plan,
            execution_history=failure_info["execution_history"],
            failure_reason=failure_info["failure_reason"],
            replan_history=failure_info["replan_history"],
            current_node_name=plan_graph.current_node_name,
        )
        self.logger.info("Calling model for replan instructions...")
        response = self._model.process(final_prompt)
        self.logger.info(f"Replan response: {response}")
        return response

    def _determine_restart_node(self, adjustments: Adjustments) -> Optional[str]:
        if adjustments.action == "replan":
            restart_node_name = adjustments.restart_node_name
        elif adjustments.action == "breakdown":
            if adjustments.new_subtasks and len(adjustments.new_subtasks) > 0:
                restart_node_name = adjustments.new_subtasks[0].name
            else:
                print("No subtasks found for breakdown action. Aborting execution.")
                return None
        else:
            self.logger.warning("Unknown action. Aborting execution.")
            return None

        if restart_node_name in self.plan_graph.nodes:
            return restart_node_name
        else:
            if restart_node_name:
                self.logger.warning(
                    f"Restart node '{restart_node_name}' does not exist. Aborting execution."
                )
            return None

    def apply_adjustments_to_plan(self, node_name: str, adjustments: Adjustments,
                                  execution_history: Steps):
        if adjustments.action == "breakdown":
            original_node = self.plan_graph.nodes.pop(node_name)
            if not original_node:
                self.logger.warning(
                    f"No original node found for Name ='{node_name}'. Skipping."
                )
                return
            new_subtasks = adjustments.new_subtasks
            if not new_subtasks:
                self.logger.warning(
                    "No 'new_subtasks' found for breakdown action. Skipping."
                )
                return
            # Insert new subtasks as nodes
            for st in new_subtasks:
                new_subtask_name = st.name
                if not new_subtask_name:
                    self.logger.warning(f"No 'name' in subtask: {st}. Skipping.")
                    continue
                self.plan_graph.add_node(st)
            # Update references to the removed node
            for nid, node in self.plan_graph.nodes.items():
                if node_name == node.next_node:
                    node.next_node = new_subtasks[0].name
        elif adjustments.action == "replan":
            restart_node_name = adjustments.restart_node_name
            modifications = (
                adjustments.modifications if adjustments.modifications else []
            )
            for mod in modifications:
                if not mod.name:
                    self.logger.warning(f"No 'Name' in modification: {mod}. Skipping.")
                    continue
                if mod.name in self.plan_graph.nodes:
                    node = self.plan_graph.nodes.pop(mod.name)
                    node.current_attempts = 0
                    node.description = (
                        mod.description if mod.description else node.description
                    )
                    node.next_node = mod.next_node if mod.next_node else node.next_node
                    node.evaluation_threshold = (
                        mod.evaluation_threshold
                        if mod.evaluation_threshold
                        else node.evaluation_threshold
                    )
                    node.max_attempts = (
                        mod.max_attempts if mod.max_attempts else node.max_attempts
                    )
                    node.category = mod.category if mod.category else node.category
                    self.plan_graph.add_node(node)
                else:
                    # If the node does not exist, create a new Node
                    self.plan_graph.add_node(mod)
            if restart_node_name and restart_node_name not in self.plan_graph.nodes:
                new_node = Node(
                    name=restart_node_name,
                    description="Automatically added restart node",
                )
                self.plan_graph.add_node(new_node)

        else:
            self.logger.warning(f"Unknown action in adjustments: {adjustments.action}")
        execution_history.adjust_plan(adjustments.action, self.plan_graph.to_plan())