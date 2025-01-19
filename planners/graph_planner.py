# planners/graph_planner.py

import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from langchain_core.tools import BaseTool
from utils.logger import get_logger
from config import Config
from validators import ScoreValidator
from utils.context_manager import ContextManager
from .generic_planner import GenericPlanner, Step
from models.model_registry import ModelRegistry


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
    DEFAULT_PROMPT = """\
{context}
Background: {background}
Now, process the above context and handle the following task:
Task Desc: {task_description}
Task Use Tool: {task_use_tool}
Task Tool Description: {tool_description}
If Task Use Tool is `False`, process according to the `Task Description`
If Task Use Tool is `True`, process according to the `Task Tool`,
For each tool argument, based on context and human's question to generate arguments value
according to the argument description.

The result must not contain any explanatory note (like '// explain'). Provide a pure JSON string that can be parsed.

Example:
the example used tool:
{
    "use_tool": true,
    "tool_name": "Event",
    "tool_arguments": {
        "eventId": "1000"
    }
}
the example used context:
{
    "use_tool": false,
    "response": "result detail"
}
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

    # Store the node's prompt in an instance variable, defaulting to DEFAULT_PROMPT
    _prompt: str = field(default=DEFAULT_PROMPT, init=False)

    @property
    def prompt(self) -> str:
        return self._prompt

    @prompt.setter
    def prompt(self, value: str):
        self._prompt = value

    def execute(
        self, model, context_manager: ContextManager, background: str = ""
    ) -> str:
        """
        Update the context with this node's info, build a prompt from context + background,
        and call model.process(...).
        """
        print(f"Executing Node {self.id}: {self.task_description}")
        self.current_attempts += 1

        tool_description = ""
        if self.task_use_tool:
            tool_description = str(self.task_tool.args_schema.model_json_schema())

        # Render final node prompt
        final_prompt = self._prompt.format(
            context=context_manager.context_to_str(),
            background=background,
            task_description=self.task_description,
            task_use_tool=self.task_use_tool,
            tool_description=tool_description,
        )

        response = model.process(final_prompt)
        cleaned = response.replace("```json", "").replace("```", "").strip()
        # Attempt JSON Parse
        try:
            data = json.loads(cleaned)
            if data["use_tool"]:
                response = (
                    f"task tool description: {self.task_tool.description}\n"
                    f"task tool response : {self.task_tool.invoke(data['tool_arguments'])}"
                )
            else:
                response = data["response"]
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.error(f"Raw LLM response was: {cleaned}")
            raise ValueError("Invalid JSON format in planner response.")

        # Add node info to context
        key = f"Node-{self.id}"

        context_manager.add_context(
            key,
            f"""
        task description: {self.task_description}
        task response: {response}
        """,
        )
        print(response)
        return response

    def validate(self, result: str, model) -> float:
        """
        Validate the node output using 'model' for the ScoreValidator.
        """
        score = evaluate_result(self.task_description, result, model)
        execution_result = ExecutionResult(
            output=result, validation_score=score, timestamp=datetime.now()
        )
        self.execution_results.append(execution_result)
        return score

    def should_replan(self) -> bool:
        if not self.execution_results:
            return False
        last_score = self.execution_results[-1].validation_score
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

    def execute_plan(
        self, model, context_manager: ContextManager, background: str = ""
    ):
        """
        Executes each node in sequence. If a node fails validation, attempt replan.
        """
        self.current_node_id = self.current_node_id or self.start_node_id
        while self.current_node_id:
            if self.current_node_id not in self.nodes:
                print(
                    f"Node {self.current_node_id} does not exist in the plan. Aborting execution."
                )
                break

            node = self.nodes[self.current_node_id]
            result = node.execute(model, context_manager, background=background)
            score = node.validate(result, model)
            print(f"Node {node.id} execution score: {score}")

            if score >= node.validation_threshold:
                # Move on
                if node.next_nodes:
                    self.current_node_id = node.next_nodes[0]
                else:
                    print("Plan execution completed successfully.")
                    break
            else:
                if node.should_replan():
                    print(f"Replanning needed at Node {node.id}")
                    failure_info = self.prepare_failure_info(node)
                    replan_response = self.call_llm_for_replan(
                        model, failure_info, context_manager
                    )
                    adjustments = parse_llm_response(replan_response)
                    if adjustments:
                        self.replan(self.current_node_id, adjustments)
                        self.current_node_id = self.determine_restart_node(adjustments)
                        if not self.current_node_id:
                            break
                    else:
                        print("Could not parse LLM response for replan, aborting.")
                        break
                else:
                    # Retry the same node
                    print(f"Retrying Node {node.id}")
                    continue

    def prepare_failure_info(self, node: Node) -> Dict:
        return {
            "failure_reason": (
                node.failed_reasons[-1] if node.failed_reasons else "Unknown"
            ),
            "execution_history": [
                {
                    "node_id": n.id,
                    "results": [er.validation_score for er in n.execution_results],
                }
                for n in self.nodes.values()
            ],
            "replan_history": self.replan_history.history,
        }

    def call_llm_for_replan(
        self, model, failure_info: Dict, context_manager: ContextManager
    ) -> str:
        plan_summary = self.summarize_plan()
        context_str = context_manager.context_to_str()

        final_prompt = self._prompt.format(
            context_str=context_str,
            plan_summary=plan_summary,
            execution_history=failure_info["execution_history"],
            failure_reason=failure_info["failure_reason"],
            replan_history=failure_info["replan_history"],
        )

        print("Calling model for replan instructions...")
        response = model.process(final_prompt)
        print("Replan response:", response)
        return response

    def summarize_plan(self) -> str:
        summary = ""
        for n in self.nodes.values():
            summary += f"Node {n.id}: {n.task_description}, Next: {n.next_nodes}\n"
        return summary

    def replan(self, node_id: str, llm_response: str):
        self.replan_history.add_record(
            {
                "timestamp": datetime.now(),
                "node_id": node_id,
                "failure_reason": (
                    self.nodes[node_id].failed_reasons[-1]
                    if self.nodes[node_id].failed_reasons
                    else "Unknown"
                ),
                "llm_response": llm_response,
            }
        )
        apply_adjustments_to_plan(self, node_id, llm_response, self.nodes)

    def determine_restart_node(self, llm_response: str) -> Optional[str]:
        adjustments = json.loads(llm_response)
        action = adjustments.get("action")
        if action == "replan":
            restart_node_id = adjustments.get("restart_node_id")
        elif action == "breakdown":
            new_subtasks = adjustments.get("new_subtasks", [])
            if new_subtasks:
                restart_node_id = new_subtasks[0].get("id")
            else:
                print("No subtasks found for breakdown action. Aborting execution.")
                return None
        else:
            print("Unknown action. Aborting execution.")
            return None

        if restart_node_id and restart_node_id in self.nodes:
            return restart_node_id
        else:
            if restart_node_id:
                print(
                    f"Restart node '{restart_node_id}' does not exist. Aborting execution."
                )
            return None


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
        knowledge: str = "",
        background: str = "",
    ) -> List[Step]:
        """
        1) Use the base class to get Steps, guided by `knowledge`.
        2) Convert them into a PlanGraph.
        3) Store `background` for later usage when executing nodes.
        4) Execute plan_graph using self.model and self.context_manager.
        5) Return Steps (for reference).
        """
        steps = super().plan(task, tools, knowledge=knowledge)
        self._background = background  # Save background for node execution

        plan_graph = PlanGraph()
        # If user gave a custom replan prompt, override PlanGraph's prompt
        if self._replan_prompt:
            plan_graph.prompt = self._replan_prompt

        previous_node = None

        tool_map = dict()
        if tools is not None:
            tool_map = {tool.name: tool for tool in tools}

        for idx, step in enumerate(steps, start=1):
            node_id = chr(65 + idx - 1)  # A, B, C, ...
            next_node_id = chr(65 + idx) if idx < len(steps) else ""

            node = Node(
                id=node_id,
                task_description=step.description,
                task_use_tool=step.use_tool,
                next_nodes=[next_node_id] if next_node_id else [],
                validation_threshold=0.8,
                max_attempts=3,
            )
            # If user gave a custom node prompt, override
            if self._node_prompt:
                node.prompt = self._node_prompt

            if node.task_use_tool and step.tool_name in tool_map:
                node.task_tool_name = step.tool_name
                node.task_tool = tool_map.get(node.task_tool_name)

            plan_graph.add_node(node)

            if previous_node:
                previous_node.set_next_node(node)
            previous_node = node

        self.plan_graph = plan_graph
        self.logger.info("PlanGraph built. Executing now...")

        # Execute the plan with background
        self.plan_graph.execute_plan(
            model=self.model,
            context_manager=self.context_manager,
            background=self._background,
        )
        return steps


#
# HELPER FUNCTIONS
#
def evaluate_result(rtask_description: str, result: str, model) -> float:
    """
    Use ScoreValidator(model) to parse and produce a numeric score [0..1].
    """
    sv = ScoreValidator(model)  # Pass the model here
    validation_str = sv.validate(rtask_description, result)
    print(validation_str)
    decision, total_score, scores = sv.parse_scored_validation_response(validation_str)
    print("\nTotal Score:", total_score)
    print("Scores by Criterion:", scores)
    print("Final Decision:", decision)

    # e.g., total_score=35 => 35/40=0.875
    return total_score / 40


def parse_llm_response(llm_response: str) -> Optional[str]:
    try:
        json.loads(llm_response)
        return llm_response
    except json.JSONDecodeError as e:
        print("Failed to parse LLM response as JSON:", e)
        return None


def apply_adjustments_to_plan(
    plan_graph, node_id: str, adjustments_str: str, nodes_dict: Dict[str, Node]
):
    adjustments = json.loads(adjustments_str)
    action = adjustments.get("action")

    if action == "breakdown":
        original_node = nodes_dict.pop(node_id, None)
        if not original_node:
            return
        new_subtasks = adjustments.get("new_subtasks", [])
        if not new_subtasks:
            return

        # Insert new subtasks as nodes
        for st in new_subtasks:
            new_node = Node(
                id=st["id"],
                task_description=st["task_description"],
                next_nodes=original_node.next_nodes,
                validation_threshold=st.get("validation_threshold", 0.8),
                max_attempts=st.get("max_attempts", 3),
            )
            plan_graph.add_node(new_node)

        # Update references to the removed node
        for nid, n in nodes_dict.items():
            if node_id in n.next_nodes:
                n.next_nodes.remove(node_id)
                if new_subtasks:
                    n.next_nodes.append(new_subtasks[0]["id"])

    elif action == "replan":
        restart_node_id = adjustments.get("restart_node_id")
        modifications = adjustments.get("modifications", [])

        for mod in modifications:
            if not isinstance(mod, dict):
                print(f"Each modification must be a dict, got: {mod}")
                continue
            if "node_id" not in mod:
                print(f"No 'node_id' in modification: {mod}")
                continue
            mod_id = mod["node_id"]
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
            else:
                new_node = Node(
                    id=mod_id,
                    task_description=mod["task_description"],
                    next_nodes=mod.get("next_nodes", []),
                    validation_threshold=mod.get("validation_threshold", 0.8),
                    max_attempts=mod.get("max_attempts", 3),
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
        print("Unknown action in adjustments.")
