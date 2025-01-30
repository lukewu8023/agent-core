# planners/graph_planner.py

import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from langchain_core.tools import BaseTool
from utils.logger import get_logger
from validators import ScoreValidator
from utils.context_manager import ContextManager
from .generic_planner import GenericPlanner, Step


@dataclass
class ExecutionResult:
    output: str
    validation_score: float
    timestamp: datetime


@dataclass
class Node:
    """
    Represents a single node in the PlanGraph.
    """

    logger = get_logger("graph-planner.node")  # minimal usage for node-level logs

    # Expose Node's default prompt for execution
    DEFAULT_PROMPT = """

<Background>
{background}
</Background>

{context}

Now, process the above context and handle the following task:

<Task>
Task Desc: {task_description}
</Task>

<Tool Use>
Task Use Tool: {task_use_tool}
Task Tool Description: {tool_description}
If Task Use Tool is `False`, process according to the `Task Description`,
If Task Use Tool is `True`, process according to the `Task Tool`,,
For each tool argument, based on context and human's question to generate arguments value according to the argument description.

The result must not contain any explanatory note (like '// explain'). Provide a pure JSON string that can be parsed.

Example:
the example used tool:
{{
    "use_tool": true,
    "tool_name": "Event",
    "tool_arguments": {{
        "eventId": "1000"
    }}
}}
the example used context:
{{
    "use_tool": false,
    "response": "result detail"
}}
</Tool Use>
"""

    id: str
    task_description: str
    next_nodes: List[str] = field(default_factory=list)

    task_use_tool: bool = False
    task_tool_name: str = ""
    task_tool: BaseTool = None

    execution_results: List[ExecutionResult] = field(default_factory=list)
    validation_threshold: float = 0.8
    max_attempts: int = 3
    current_attempts: int = 0
    failed_reasons: List[str] = field(default_factory=list)

    task_category: str = "default"

    # Store the node's prompt in an instance variable, defaulting to DEFAULT_PROMPT
    _prompt: str = field(default=DEFAULT_PROMPT, init=False)

    @property
    def prompt(self) -> str:
        return self._prompt

    @prompt.setter
    def prompt(self, value: str):
        self._prompt = value

    def should_replan(self) -> bool:
        if not self.execution_results:
            return False
        # The last item in execution_results might be a str or an ExecutionResult
        last_entry = self.execution_results[-1]
        if isinstance(last_entry, ExecutionResult):
            last_score = last_entry.validation_score
        else:
            # If it's a raw string, no numeric score is available
            return False

        if last_score >= self.validation_threshold:
            return False
        elif self.current_attempts >= self.max_attempts:
            failure_reason = (
                f"Failed to reach threshold after {self.max_attempts} attempts."
            )
            self.failed_reasons.append(failure_reason)
            return True
        return False

    def set_next_node(self, node: "Node"):
        if node.id not in self.next_nodes:
            self.next_nodes.append(node.id)


@dataclass
class ReplanHistory:
    history: List[Dict] = field(default_factory=list)

    def add_record(self, record: Dict):
        self.history.append(record)


@dataclass
class PlanGraph:
    """
    Holds multiple Node objects in a directed structure.
    Executes them in order, and triggers replan logic if needed.
    """

    logger = get_logger("graph-planner")

    # The PlanGraph has one main replan prompt
    DEFAULT_PROMPT = """\
{context_str}

Below is the current plan and its execution state:

**Plan Summary:**
{plan_summary}

**Execution History:**
{execution_history}

**Failure Reason:**
{failure_reason}

**Replanning History:**
{replan_history}

Instructions:
- Decide if we do a "breakdown" or "replan".
- Return valid JSON with keys: "action", "new_subtasks", "restart_node_id", "modifications", "rationale".
"""

    nodes: Dict[str, Node] = field(default_factory=dict)
    start_node_id: Optional[str] = None
    replan_history: ReplanHistory = field(default_factory=ReplanHistory)
    current_node_id: Optional[str] = None

    # We'll store the replan prompt in an instance variable
    _prompt: str = field(default=DEFAULT_PROMPT, init=False)

    @property
    def prompt(self) -> str:
        """Prompt for replan instructions in call_llm_for_replan."""
        return self._prompt

    @prompt.setter
    def prompt(self, value: str):
        self._prompt = value

    def add_node(self, node: Node):
        self.nodes[node.id] = node
        if self.start_node_id is None:
            self.start_node_id = node.id

    def summarize_plan(self) -> str:
        summary = ""
        for n in self.nodes.values():
            summary += f"Node {n.id}: {n.task_description}, Next: {n.next_nodes}\n"
        return summary


class GraphPlanner(GenericPlanner):
    """
    Extends GenericPlanner but builds a PlanGraph, uses context, and
    executes node-based logic with re-planning.
    """

    def __init__(
        self,
        model: str = None,
        log_level: Optional[str] = None,
        prompt: str = None,
        node_prompt: str = None,
        replan_prompt: str = None,
    ):
        """
        :param prompt:       Override for the base planning prompt inherited from GenericPlanner.
        :param node_prompt:  Override for Node execution prompt.
        :param replan_prompt:Override for PlanGraph's replan prompt.
        """
        super().__init__(model=model, log_level=log_level, prompt=prompt)
        self.logger = get_logger("graph-planner", log_level)
        self.plan_graph: Optional[PlanGraph] = None
        self.context_manager = ContextManager()
        self._background = ""  # We'll store background for node execution

        self._node_prompt = node_prompt
        self._replan_prompt = replan_prompt

    def plan(
        self,
        task: str,
        tools: Optional[List[BaseTool]],
        execute_history: list = None,
        knowledge: str = "",
        background: str = "",
        categories: Optional[List[str]] = None,
        agent=None,
    ) -> List[Step]:
        """
        1) Use the base class to get Steps, guided by `knowledge`.
        2) Convert them into a PlanGraph.
        3) Store `background` for later usage when executing nodes.
        4) Return Steps (for reference).
        """
        steps = super().plan(
            task, tools, execute_history, knowledge, background, categories, agent
        )
        self._background = background  # Save background for node execution

        plan_graph = PlanGraph()
        # If user gave a custom replan prompt, override PlanGraph's prompt
        if self._replan_prompt:
            plan_graph.prompt = self._replan_prompt

        previous_node = None
        tool_map = {}
        if tools is not None:
            tool_map = {tool.name: tool for tool in tools}

        for idx, step in enumerate(steps, start=1):
            node_id = chr(65 + idx - 1)  # A, B, C, ...
            next_node_id = chr(65 + idx) if idx < len(steps) else ""

            node = Node(
                id=node_id,
                task_description=step.description,
                task_use_tool=step.use_tool,
                task_tool_name=step.tool_name,
                task_tool=tool_map.get(step.tool_name) if step.tool_name else None,
                next_nodes=[next_node_id] if next_node_id else [],
                validation_threshold=0.8,
                max_attempts=3,
                task_category=step.category,  # tie the step's category to the node
            )
            # If user gave a custom node prompt, override
            if self._node_prompt:
                node.prompt = self._node_prompt

            plan_graph.add_node(node)

            if previous_node:
                previous_node.set_next_node(node)
            previous_node = node

        self.plan_graph = plan_graph
        return steps

    def execute_plan(
        self,
        execute_history: list,
        agent=None,
    ):
        """
        Executes each node in sequence from plan_graph.
        If a node fails validation, attempt replan or re-try.
        """
        if not self.plan_graph:
            self.logger.error("No plan_graph found. Did you call plan() first?")
            return

        pg = self.plan_graph
        pg.current_node_id = pg.current_node_id or pg.start_node_id
        while pg.current_node_id:
            if pg.current_node_id not in pg.nodes:
                self.logger.error(
                    f"Node {pg.current_node_id} does not exist in the plan. Aborting execution."
                )
                break

            node = pg.nodes[pg.current_node_id]
            result = self._execute_node(node)
            score = self._validate_node(node, result, agent)
            self.logger.info(f"Node {node.id} execution score: {score}")

            if score >= node.validation_threshold:
                node.result = result
                execute_history.append(
                    {
                        "step_name": node.id,
                        "step_description": node.task_description,
                        "step_result": str(result),
                    }
                )
                # Move on
                if node.next_nodes:
                    pg.current_node_id = node.next_nodes[0]
                else:
                    self.logger.info("Plan execution completed successfully.")
                    break
            else:
                if node.should_replan():
                    self.logger.warning(f"Replanning needed at Node {node.id}")
                    failure_info = self.prepare_failure_info(node)
                    replan_response = self.call_llm_for_replan(pg, failure_info)
                    adjustments = parse_llm_response(replan_response)
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
                        apply_adjustments_to_plan(
                            self.plan_graph, node.id, adjustments, pg.nodes
                        )
                        restart_node_id = self.determine_restart_node(adjustments)
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
                    continue

        return "Task execution completed using GraphPlanner."

    #
    # The node-level methods, previously in Node
    #
    def _execute_node(self, node: Node) -> str:
        """
        Build prompt + call the LLM. If 'use_tool', invoke the tool.
        """
        print(f"Executing Node {node.id}: {node.task_description}")
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

        final_prompt = node.prompt.format(
            context=self.context_manager.context_to_str(),
            background=self._background,
            task_description=node.task_description,
            task_use_tool=node.task_use_tool,
            tool_description=tool_description,
        )

        response = self.model.process(final_prompt)
        cleaned = response.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
            if data["use_tool"]:
                if node.task_tool is not None:
                    response = (
                        f"task tool description: {node.task_tool.description}\n"
                        f"task tool response : {node.task_tool.invoke(data['tool_arguments'])}"
                    )
                else:
                    response = "Tool usage was requested, but no tool is attached to this node."
            else:
                response = data["response"]
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.error(f"Raw LLM response was: {cleaned}")
            raise ValueError("Invalid JSON format in planner response.")

        # Add node info to context
        key = f"Node-{node.id}"
        self.context_manager.add_context(
            key,
            f"""
Task description: {node.task_description}
Task response: {response}
        """,
        )
        # Keep the raw response in node's execution_results for reference
        node.execution_results.append(response)
        print(response)
        return response

    def _validate_node(self, node: Node, result: str, agent) -> float:
        """
        Validate the node output using agent's validator if enabled.
        Return 0..1 scale.
        """
        if not agent or not agent.validators_enabled:
            # Validation is disabled; treat as success
            execution_result = ExecutionResult(
                output=result, validation_score=1.0, timestamp=datetime.now()
            )
            node.execution_results.append(execution_result)
            return 1.0

        chosen_cat = (
            node.task_category if node.task_category in agent.validators else "default"
        )
        validator = agent.validators.get(chosen_cat)
        if not validator:
            self.logger.warning(
                f"No validator found for category '{chosen_cat}'. Validation skipped."
            )
            execution_result = ExecutionResult(
                output=result, validation_score=1.0, timestamp=datetime.now()
            )
            node.execution_results.append(execution_result)
            return 1.0

        decision, score, details = validator.validate(node.task_description, result)
        if decision == "Rerun Subtask":
            numeric_score = 0.0
        else:
            numeric_score = float(score) / 40.0

        execution_result = ExecutionResult(
            output=result, validation_score=numeric_score, timestamp=datetime.now()
        )
        node.execution_results.append(execution_result)
        return numeric_score

    #
    # Replan-related helpers
    #
    def prepare_failure_info(self, node: Node) -> Dict:
        """
        Produce the context for replan prompt.
        """
        pg = self.plan_graph
        return {
            "failure_reason": (
                node.failed_reasons[-1] if node.failed_reasons else "Unknown"
            ),
            "execution_history": [
                {
                    "node_id": n.id,
                    "results": [
                        er.validation_score if isinstance(er, ExecutionResult) else None
                        for er in n.execution_results
                    ],
                }
                for n in pg.nodes.values()
            ],
            "replan_history": pg.replan_history.history,
        }

    def call_llm_for_replan(self, plan_graph: PlanGraph, failure_info: Dict) -> str:
        plan_summary = plan_graph.summarize_plan()
        context_str = self.context_manager.context_to_str()

        final_prompt = plan_graph.prompt.format(
            context_str=context_str,
            plan_summary=plan_summary,
            execution_history=failure_info["execution_history"],
            failure_reason=failure_info["failure_reason"],
            replan_history=failure_info["replan_history"],
        )

        print("Calling model for replan instructions...")
        response = self.model.process(final_prompt)
        print("Replan response:", response)
        return response

    def determine_restart_node(self, llm_response: str) -> Optional[str]:
        adjustments = json.loads(llm_response)
        action = adjustments.get("action")
        if action == "replan":
            restart_node_id = adjustments.get("restart_node_id")
        elif action == "breakdown":
            new_subtasks = adjustments.get("new_subtasks", [])
            if new_subtasks:
                restart_node_id = next(
                    (st.get("id") for st in new_subtasks if "id" in st), None
                )
                if not restart_node_id:
                    print(
                        "No valid 'id' found in new_subtasks, cannot restart. Aborting."
                    )
                    return None
            else:
                print("No subtasks found for breakdown action. Aborting execution.")
                return None
        else:
            print("Unknown action. Aborting execution.")
            return None

        if restart_node_id and restart_node_id in self.plan_graph.nodes:
            return restart_node_id
        else:
            if restart_node_id:
                print(
                    f"Restart node '{restart_node_id}' does not exist. Aborting execution."
                )
            return None


#
# HELPER FUNCTIONS
#
# def evaluate_result(rtask_description: str, result: str, model) -> float:
#     """
#     Use ScoreValidator(model) to parse and produce a numeric score [0..1].
#     """
#     sv = ScoreValidator(model)  # Pass the model here
#     validation_str = sv.validate(rtask_description, result)
#     print(validation_str)
#     decision, total_score, scores = sv.parse_scored_validation_response(validation_str)
#     print("\nTotal Score:", total_score)
#     print("Scores by Criterion:", scores)
#     print("Final Decision:", decision)

#     # e.g., total_score=35 => 35/40=0.875
#     return total_score / 40


def parse_llm_response(llm_response: str) -> Optional[str]:
    try:
        json.loads(llm_response)
        return llm_response
    except json.JSONDecodeError as e:
        print("Failed to parse LLM response as JSON:", e)
        return None


def apply_adjustments_to_plan(
    plan_graph, node_id: str, adjustments_str: str, nodes_dict: Dict[str, "Node"]
):
    adjustments = json.loads(adjustments_str)
    action = adjustments.get("action")

    if action == "breakdown":
        original_node = nodes_dict.pop(node_id, None)
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
            subtask_id = st.get("id")
            if not subtask_id:
                plan_graph.logger.warning(f"No 'id' in subtask: {st}. Skipping.")
                continue

            task_description = st.get("task_description", "No description provided.")
            new_node = Node(
                id=subtask_id,
                task_description=task_description,
                next_nodes=original_node.next_nodes,
                validation_threshold=st.get("validation_threshold", 0.8),
                max_attempts=st.get("max_attempts", 3),
                task_category=st.get("step_category", "default"),
            )
            plan_graph.add_node(new_node)

        # Update references to the removed node
        for nid, n in nodes_dict.items():
            if node_id in n.next_nodes:
                n.next_nodes.remove(node_id)
                if new_subtasks:
                    # We link to the first newly created node only if it had an 'id'
                    first_valid = next(
                        (sub["id"] for sub in new_subtasks if "id" in sub), None
                    )
                    if first_valid:
                        n.next_nodes.append(first_valid)

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

            if mod_id in nodes_dict:
                node = nodes_dict[mod_id]
                node.task_description = mod.get(
                    "task_description", node.task_description
                )
                node.next_nodes = mod.get("next_nodes", node.next_nodes)
                node.validation_threshold = mod.get(
                    "validation_threshold", node.validation_threshold
                )
                node.max_attempts = mod.get("max_attempts", node.max_attempts)
                node.task_category = mod.get("step_category", node.task_category)
            else:
                # If the node does not exist, create a new Node
                new_description = mod.get("task_description", "No description")
                new_node = Node(
                    id=mod_id,
                    task_description=new_description,
                    next_nodes=mod.get("next_nodes", []),
                    validation_threshold=mod.get("validation_threshold", 0.8),
                    max_attempts=mod.get("max_attempts", 3),
                    task_category=mod.get("step_category", "default"),
                )
                plan_graph.add_node(new_node)

        if restart_node_id and restart_node_id not in nodes_dict:
            new_node = Node(
                id=restart_node_id,
                task_description="Automatically added restart node",
                next_nodes=[],
                validation_threshold=0.8,
                max_attempts=3,
            )
            plan_graph.add_node(new_node)

    else:
        plan_graph.logger.warning(f"Unknown action in adjustments: {action}")
