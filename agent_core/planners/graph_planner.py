# planners/graph_planner.py

import json
import logging
import re

from agent_core.evaluators import BaseEvaluator
from agent_core.planners.base_planner import (
    BasePlanner,
    tool_knowledge_format,
)
from agent_core.planners.generic_planner import GenericPlanner, Step
from agent_core.models.model_registry import ModelRegistry
from agent_core.utils.context_manager import ContextManager
from agent_core.entities.steps import Steps
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from langchain_core.tools import BaseTool

from agent_core.utils.llm_chat import LLMChat
from agent_core.utils.logger import get_logger


@dataclass
class ExecutionResult:
    output: str
    evaluation_score: float
    timestamp: datetime


@dataclass
class ReplanHistory:
    history: List[Dict] = field(default_factory=list)

    def add_record(self, record: Dict):
        self.history.append(record)


@dataclass
class Node:
    """
    Represents a single node in the PlanGraph.
    """

    id: str
    task_description: str
    next_nodes: List[str] = field(default_factory=list)

    task_use_tool: bool = False
    task_tool_name: str = ""
    task_tool: BaseTool = None

    execution_results: List[ExecutionResult] = field(default_factory=list)
    evaluation_threshold: float = 0.9
    max_attempts: int = 3
    current_attempts: int = 0
    failed_reasons: List[str] = field(default_factory=list)

    task_category: str = "default"

    def set_next_node(self, node: "Node"):
        if node.id not in self.next_nodes:
            self.next_nodes.append(node.id)


@dataclass
class PlanGraph:
    """
    Holds multiple Node objects in a directed structure.
    Node-based plan execution with possible replan logic.
    """

    logger = get_logger("plan-graph")

    nodes: Dict[str, Node] = field(default_factory=dict)
    start_node_id: Optional[str] = None
    replan_history: ReplanHistory = field(default_factory=ReplanHistory)
    current_node_id: Optional[str] = None

    def add_node(self, node: Node):
        self.nodes[node.id] = node
        if self.start_node_id is None:
            self.start_node_id = node.id

    def summarize_plan(self) -> str:
        summary = ""
        for n in self.nodes.values():
            summary += f"Node {n.id}: {n.task_description}, Next: {n.next_nodes}\n"
        return summary


def _should_replan(node: Node) -> bool:
    """
    Checks if the node's last result is below threshold and attempts are exceeded.
    """
    if not node.execution_results:
        return False
    last_entry = node.execution_results[-1]
    if isinstance(last_entry, ExecutionResult):
        last_score = last_entry.evaluation_score
    else:
        return False

    if last_score >= node.evaluation_threshold:
        return False
    elif node.current_attempts >= node.max_attempts:
        failure_reason = f"Node {node.id} failed to reach threshold after {node.max_attempts} attempts."
        node.failed_reasons.append(failure_reason)
        return True
    return False


class GraphPlanner(BasePlanner):
    """
    A planner that builds a PlanGraph, uses context, and executes node-based logic with re-planning.
    Inherits from BasePlanner instead of GenericPlanner.
    """

    DEFAULT_EXECUTE_PROMPT = """

<Background>
{background}
</Background>

{context}

Now, based on the above background and context, process the following task, please notice not to repeat those responses in the failed attempts and take those suggestions before respond.

<Task>
<Root Task>
{task}
</Root Task>
<Current Task>
{task_description}
</Current Task>
</Task>

Task Use Tool: {task_use_tool}

<Tool Use>
Task Tool Description: {tool_description}
If Task Use Tool is `False`, process according to the description of the current task,
If Task Use Tool is `True`, process using tools,
For each tool argument, based on context and human's question to generate arguments value according to the argument description.

The result must not contain any explanatory note (like '// explain'). Provide a pure JSON string that can be parsed.
</Tool Use>

<Output Example>
If task use tool is true, example:
{{
    "use_tool": true,
    "tool_name": "Event",
    "tool_arguments": {{
        "eventId": "1000"
    }}
}}
If task use tool is false, example:
{{
    "use_tool": false,
    "response": "result detail"
}}
</Output Example>
"""

    DEFAULT_REPLAN_PROMPT = """
You are an intelligent assistant helping to adjust a task execution plan represented as a graph of subtasks. Below are the details:

**Background:**
{background}

**Knowledge:**
{knowledge}

**Tools:**
{tools_knowledge}

**Root Task:**
{root_task}

**Categories:**
{categories_str}

**Current Plan:**
{plan_summary}

**Execution History:**
{execution_history}
(Notes: 1.0 is the full score. The closer to 1.0, the closer to accuracy. 0.9 is the threshold. Less than 0.9 is failed.)

**Failure Reason:**
{failure_reason}

**Replanning History:**
{replan_history}

**Instructions:**
- Analyze the Current Plan, Execution History, Failure Reason and Replanning History to decide on one of two actions:
    1. **breakdown**: Break down the task of failed node {current_node_id} into smaller subtasks.
    2. **replan**: Go back to a previous node for replanning, 
- If you choose **breakdown**, provide detailed descriptions of the new subtasks, only breakdown the current (failed) node, otherwise it should be replan. ex: if current node is B, breakdown nodes should be B.1, B.2, if current node is B.2, breakdown nodes should be B.2.1, B.2.2... and make the all nodes as chain eventually.
- If you choose **replan**, specify which node to return to and suggest any modifications to the plan after that node, do not repeat previous failure replanning in the Replanning History.
- The id generated following the naming convention as A.1, B.1.2, C.2.5.2, new id (not next_nodes) generation example: current: B > new sub: B.1, current: B.2.2.2 > new sub: B.2.2.2.1
- Return your response in the following JSON format (do not include any additional text):

```json
{{
    "action": "breakdown" or "replan",
    "new_subtasks": [  // Required if action is "breakdown"
        {{
            "id": "unique_task_id",
            "task_description": "Description of the subtask",
            "next_nodes": ["next_node_id_1", "next_node_id_2"],
            "evaluation_threshold": 0.9,
            "max_attempts": 3
        }}
    ],
    "restart_node_id": "node_id",  // Required if action is "replan"
    "modifications": [  // Optional, used if action is "replan"
        {{
            "node_id": "node_to_modify_id",
            "task_description": "Modified description",
            "next_nodes": ["next_node_id_1", "next_node_id_2"],
            "evaluation_threshold": 0.9,
            "max_attempts": 3
        }}
    ],
    "rationale": "Explanation of your reasoning here"
}}

"""

    def __init__(self, model_name: str = None, log_level: Optional[str] = None):
        super().__init__(model_name, log_level)
        self.plan_graph: Optional[PlanGraph] = None
        self.context_manager = ContextManager()

        self._replan_prompt = self.DEFAULT_REPLAN_PROMPT
        self._execute_prompt = self.DEFAULT_EXECUTE_PROMPT

    @property
    def replan_prompt(self) -> str:
        """Prompt for replan instructions in call_llm_for_replan."""
        return self._replan_prompt

    @replan_prompt.setter
    def replan_prompt(self, value: str):
        self._replan_prompt = value

    @property
    def execute_prompt(self) -> str:
        """Prompt for node prompt"""
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
        self.logger.info(f"GraphPlanner: Creating plan for task: {task}")

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

        plan_graph.prompt = self._replan_prompt  # If needed for replan calls
        plan_graph.background = background
        plan_graph.knowledge = knowledge
        plan_graph.categories = categories
        plan_graph.task = task
        plan_graph.tools = tool_knowledge_format(tools)

        previous_node = None
        tool_map = {}
        if tools is not None:
            tool_map = {tool.name: tool for tool in tools}

        for idx, step in enumerate(plan.steps, start=1):
            node_id = chr(65 + idx - 1)  # e.g., A, B, C...
            next_node_id = chr(65 + idx) if idx < len(plan.steps) else ""

            node = Node(
                id=node_id,
                task_description=step.description,
                task_use_tool=step.use_tool,
                task_tool_name=step.tool_name,
                task_tool=tool_map.get(step.tool_name) if step.tool_name else None,
                next_nodes=[next_node_id] if next_node_id else [],
                max_attempts=3,
                task_category=step.category,
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
        evaluators_enabled: bool,
        evaluators: dict,
        context_manager: ContextManager = None,
        background: str = "",
    ):
        """
        Executes the PlanGraph node by node.
        'steps' is ignored in practice, because we use self.plan_graph.
        This signature is here for consistency with the BasePlanner interface.
        """
        execution_history = Steps()
        if not self.plan_graph:
            self.logger.error(
                "No plan graph found. Need to generate plan graph by plan() first."
            )
            return

        if context_manager:
            self.context_manager = context_manager

        pg = self.plan_graph
        pg.current_node_id = pg.current_node_id or pg.start_node_id
        while pg.current_node_id:
            if pg.current_node_id not in pg.nodes:
                self.logger.error(
                    f"Node {pg.current_node_id} does not exist in the plan. Aborting execution."
                )
                break

            node = pg.nodes[pg.current_node_id]
            response = self._execute_node(node, self.model_name, task, background)
            execution_result, details = self._evaluate_node(
                node,
                task,
                response,
                evaluators_enabled,
                evaluators,
                background,
                context_manager,
            )
            self.logger.info(
                f"Node {node.id} execution score: {execution_result.evaluation_score}"
            )

            if execution_result.evaluation_score >= node.evaluation_threshold:
                if self.context_manager:
                    attempt = "|".join(str(i) for i in range(node.current_attempts + 1))
                    self.context_manager.context = {
                        k: v
                        for k, v in self.context_manager.context.items()
                        if not re.match(
                            f"Previous Step {node.id}(.([0-9])*)* Failed Attempt ({attempt})?",
                            k,
                        )
                    }

                    # Add node info to context
                    key = f"Previous Step {node.id}"
                    if self.context_manager:
                        self.context_manager.add_context(
                            key,
                            f"""
                            Task description: {node.task_description}
                            Task response: {response}
                            """,
                        )
                    # Keep the raw response in node's execution_results for reference
                    node.execution_results.append(response)

                node.result = response
                execution_history.add_step(
                    Step(
                        name=node.id,
                        description=node.task_description,
                        result=str(response),
                    )
                )
                if node.next_nodes:
                    pg.current_node_id = node.next_nodes[0]
                else:
                    self.logger.info("Plan execution completed successfully.")
                    break
            else:
                if _should_replan(node):

                    attempt = "|".join(str(i) for i in range(node.current_attempts + 1))
                    self.context_manager.context = {
                        k: v
                        for k, v in self.context_manager.context.items()
                        if not re.match(
                            f"Previous Step {node.id}(.([0-9])*)* Failed Attempt ({attempt})?",
                            k,
                        )
                    }

                    self.logger.warning(f"Replanning needed at Node {node.id}")
                    failure_info = self.prepare_failure_info(node, details)
                    replan_response = self.call_llm_for_replan(pg, failure_info)
                    adjustments = LLMChat(self.model_name).parse_llm_response(
                        replan_response
                    )
                    if adjustments:
                        pg.replan_history.add_record(
                            {
                                "timestamp": datetime.now(),
                                "node_id": node.id,
                                "failure_reason": (
                                    node.failed_reasons[-1]
                                    if node.failed_reasons
                                    else "Unknown"
                                ),
                                "llm_response": adjustments,
                            }
                        )
                        apply_adjustments_to_plan(self.plan_graph, node.id, adjustments)
                        self.logger.info(
                            f"New plan after adjusted: {self.plan_graph.nodes}"
                        )
                        restart_node_id = self.determine_restart_node(adjustments)
                        self.cleanup_context(
                            pg.current_node_id, adjustments.get("restart_node_id")
                        )
                        if restart_node_id:
                            pg.current_node_id = restart_node_id
                        else:
                            break
                    else:
                        self.logger.error(
                            "Could not parse LLM response for replan, aborting."
                        )
                        break
                else:
                    # Retry the same node
                    self.logger.warning(f"Retrying Node {node.id}")
                    # remove node info from context
                    key = f"Previous Step {node.id}"
                    if self.context_manager:
                        self.context_manager.remove_context(key)
                        key = f"Previous Step {node.id} Failed Attempt {node.current_attempts}"
                        self.context_manager.add_context(
                            key,
                            f"""
                            Task response: {response}
                            """,
                        )

                    continue
        logging.info("Task execution completed using GraphPlanner")
        return execution_history

    def _execute_node(
        self, node: Node, model_name: str, task: str, background: str
    ) -> str:
        """
        Build prompt + call the LLM. If 'use_tool', invoke the tool.
        """
        self.logger.info(f"Executing Node {node.id}: {node.task_description}")
        node.current_attempts += 1

        tool_description = ""
        if node.task_use_tool:
            if node.task_tool is None:
                self.logger.warning(
                    f"Node {node.id} indicates 'use_tool' but 'task_tool' is None. Skipping tool usage details."
                )
            elif hasattr(node.task_tool, "args_schema"):
                tool_description = str(node.task_tool.args_schema.model_json_schema())
            else:
                self.logger.warning(
                    f"Node {node.id} indicates 'use_tool' but the provided tool lacks 'args_schema'."
                )
                tool_description = f"[Tool: {node.task_tool_name}]"

        # Node doesn't store a custom prompt, so we use self._execute_prompt
        final_prompt = self._execute_prompt.format(
            context=self.context_manager.context_to_str(),
            task=task,
            background=background,
            task_description=f"<Step {node.id}>\nTask Desc: {node.task_description}\n</Step {node.id}>",
            task_use_tool=node.task_use_tool,
            tool_description=tool_description,
        )

        response = ModelRegistry.get_model(model_name).process(final_prompt)
        cleaned = response.replace("```json", "").replace("```", "").strip()
        cleaned = cleaned.replace("\\", "\\\\")
        try:
            data = json.loads(cleaned)
            if "use_tool" in data:
                if data["use_tool"]:
                    if node.task_tool is not None:
                        try:
                            tool_response = node.task_tool.invoke(
                                data["tool_arguments"]
                            )
                            response = (
                                f"task tool description: {node.task_tool.description}\n"
                                f"task tool response : {tool_response}"
                            )
                        except Exception as e:
                            response = "Incorrect tool arguments and unexpected result when invoke the tool."
                    else:
                        response = "Tool usage was requested, but no tool is attached to this node."
                else:
                    response = data["response"]
            else:
                response = "Incorrect and unexpected structure in response."
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.error(f"Raw LLM response was: {cleaned}")
            raise ValueError("Invalid JSON format in planner response.")

        self.logger.info(f"Response:\n {response}")
        return response

    def _evaluate_node(
        self,
        node: Node,
        root_task: str,
        result: str,
        evaluators_enabled: bool,
        evaluators: Dict[str, BaseEvaluator],
        background: str,
        context_manager: ContextManager,
    ):
        """
        evaluate the node output using agent's evaluator if enabled.
        Return 0..1 scale.
        """
        if not evaluators_enabled:
            execution_result = ExecutionResult(
                output=result, evaluation_score=1.0, timestamp=datetime.now()
            )
            node.execution_results.append(execution_result)
            return execution_result, ""

        chosen_cat = (
            node.task_category if node.task_category in evaluators else "default"
        )
        evaluator = evaluators.get(chosen_cat)
        if not evaluator:
            self.logger.warning(
                f"No evaluator found for category '{chosen_cat}'. evaluation skipped."
            )
            execution_result = ExecutionResult(
                output=result, evaluation_score=1.0, timestamp=datetime.now()
            )
            node.execution_results.append(execution_result)
            return execution_result

        evaluator_result = evaluator.evaluate(
            root_task, node.task_description, result, background, context_manager
        )
        numeric_score = float(evaluator_result.score) / 40.0
        execution_result = ExecutionResult(
            output=result, evaluation_score=numeric_score, timestamp=datetime.now()
        )
        node.execution_results.append(execution_result)
        return execution_result, evaluator_result.details

    def prepare_failure_info(self, node: Node, details: str) -> Dict:
        """
        Produce the context for replan prompt.
        """
        pg = self.plan_graph
        return {
            "failure_reason": (
                node.failed_reasons[-1] if node.failed_reasons else "No"
            ),
            "execution_history": [
                {
                    "node_id": n.id,
                    "results": [
                        er.evaluation_score if isinstance(er, ExecutionResult) else None
                        for er in n.execution_results
                    ],
                }
                for n in pg.nodes.values()
            ],
            "replan_history": pg.replan_history.history,
            "evaluator": details,
        }

    def call_llm_for_replan(self, plan_graph: PlanGraph, failure_info: Dict) -> str:
        plan_summary = plan_graph.summarize_plan()
        context_str = (
            self.context_manager.context_to_str() if self.context_manager else ""
        )

        final_prompt = plan_graph.prompt.format(
            background=plan_graph.background,
            knowledge=plan_graph.knowledge,
            tools_knowledge=plan_graph.tools,
            root_task=plan_graph.task,
            context_str=context_str,
            categories_str=(
                ", ".join(plan_graph.categories)
                if plan_graph.categories
                else "(Not defined)"
            ),
            plan_summary=plan_summary,
            execution_history=failure_info["execution_history"],
            failure_reason=failure_info["failure_reason"],
            replan_history=failure_info["replan_history"],
            current_node_id=plan_graph.current_node_id,
        )

        self.logger.info("Calling model for replan instructions...")
        response = self._model.process(final_prompt)
        self.logger.info(f"Replan response: {response}")
        return response

    def determine_restart_node(self, adjustments: str) -> Optional[str]:
        # adjustments = json.loads(llm_response)
        action = adjustments.get("action")
        if action == "replan":
            restart_node_id = adjustments.get("restart_node_id")
        elif action == "breakdown":
            if (
                adjustments.get("new_subtasks")
                and len(adjustments.get("new_subtasks")) > 0
            ):
                restart_node_id = adjustments.get("new_subtasks")[0].get("id")
            else:
                print("No subtasks found for breakdown action. Aborting execution.")
                return None
            # new_subtasks = adjustments.get("new_subtasks", [])
            # if new_subtasks:
            #     restart_node_id = next(
            #         (st.get("id") for st in new_subtasks if "id" in st), None
            #     )
            #     if not restart_node_id:
            #         self.logger.warning(
            #             "No valid 'id' found in new_subtasks, cannot restart. Aborting."
            #         )
            #         return None
            # else:
            #     self.logger.warning(
            #         "No subtasks found for breakdown action. Aborting execution."
            #     )
            #     return None
        else:
            self.logger.warning("Unknown action. Aborting execution.")
            return None

        if restart_node_id in self.plan_graph.nodes:
            return restart_node_id
        else:
            if restart_node_id:
                self.logger.warning(
                    f"Restart node '{restart_node_id}' does not exist. Aborting execution."
                )
            return None

    def cleanup_context(self, current_node_id, restart_node_id):
        input_text = self.context_manager.context_to_str()
        nodes_to_remove = self.context_manager.identify_context_key(
            input_text, current_node_id, restart_node_id
        )
        for node in nodes_to_remove:
            self.context_manager.remove_context(node)


def apply_adjustments_to_plan(
    plan_graph: PlanGraph,
    node_id: str,
    adjustments: str,
):
    action = adjustments.get("action")

    if action == "breakdown":
        original_node = plan_graph.nodes.pop(node_id)
        if not original_node:
            plan_graph.logger.warning(
                f"No original node found for ID='{node_id}'. Skipping."
            )
            return
        new_subtasks = adjustments.get("new_subtasks", [])
        if not new_subtasks:
            plan_graph.logger.warning(
                "No 'new_subtasks' found for breakdown action. Skipping."
            )
            return
        # Insert new subtasks as nodes
        for st in new_subtasks:
            # original_sub_id = st.get("id") or "subtask"
            new_subtask_id = st.get("id")
            if not new_subtask_id:
                plan_graph.logger.warning(f"No 'id' in subtask: {st}. Skipping.")
                continue
            # if i < len(new_subtasks) - 1:
            #     subtask_next = [new_subtasks[i + 1]["id"]]
            # else:
            #     subtask_next = original_node.next_nodes
            new_node = Node(
                id=new_subtask_id,
                task_description=st.get("task_description", "No description provided."),
                next_nodes=st.get("next_nodes", []),
                evaluation_threshold=st.get("evaluation_threshold", 0.9),
                max_attempts=st.get("max_attempts", 3),
                task_category=st.get("step_category", "default"),
            )
            plan_graph.add_node(new_node)

        # Update references to the removed node
        for nid, node in plan_graph.nodes.items():
            if node_id in node.next_nodes:
                node.next_nodes.remove(node_id)
                node.next_nodes.append(new_subtasks[0]["id"])

    elif action == "replan":
        restart_node_id = adjustments.get("restart_node_id")
        modifications = adjustments.get("modifications", [])

        for mod in modifications:
            if not isinstance(mod, dict):
                plan_graph.logger.warning(
                    f"Modification is not a dict: {mod}. Skipping."
                )
                continue

            mod_id = mod.get("node_id")
            if not mod_id:
                plan_graph.logger.warning(
                    f"No 'node_id' in modification: {mod}. Skipping."
                )
                continue

            if mod_id in plan_graph.nodes:
                node = plan_graph.nodes.pop(mod_id)
                node.current_attempts = 0
                node.task_description = mod.get(
                    "task_description", node.task_description
                )
                node.next_nodes = mod.get("next_nodes", node.next_nodes)
                node.evaluation_threshold = mod.get(
                    "evaluation_threshold", node.evaluation_threshold
                )
                node.max_attempts = mod.get("max_attempts", node.max_attempts)
                node.task_category = mod.get("step_category", node.task_category)
                plan_graph.add_node(node)
            else:
                # If the node does not exist, create a new Node
                new_description = mod.get("task_description", "No description")
                new_node = Node(
                    id=mod_id,
                    task_description=new_description,
                    next_nodes=mod.get("next_nodes", []),
                    evaluation_threshold=mod.get("evaluation_threshold", 0.9),
                    max_attempts=mod.get("max_attempts", 3),
                    task_category=mod.get("step_category", "default"),
                )
                plan_graph.add_node(new_node)

        if restart_node_id and restart_node_id not in plan_graph.nodes:
            new_node = Node(
                id=restart_node_id,
                task_description="Automatically added restart node",
                next_nodes=[],
                evaluation_threshold=0.9,
                max_attempts=3,
            )
            plan_graph.add_node(new_node)

    else:
        plan_graph.logger.warning(f"Unknown action in adjustments: {action}")