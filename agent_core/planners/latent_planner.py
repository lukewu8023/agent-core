# planners/latent_planner.py

import json
import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any

from agent_core.utils.logger import get_logger
from agent_core.utils.context_manager import ContextManager
from agent_core.models.model_registry import ModelRegistry
from agent_core.evaluators import BaseEvaluator


@dataclass
class LatentTaskStep:
    """
    Represents a single step in the plan, with a hidden latent vector for repeated (recurrent) reasoning.
    """

    name: str
    description: str
    use_tool: bool = False
    tool_name: Optional[str] = None
    latent_vector: Optional[np.ndarray] = None

    # For demonstration: if the step depends on some external info,
    # we list the keys of the info needed
    info_required: List[str] = field(default_factory=list)

    # Execution details
    max_attempts: int = 2
    current_attempts: int = 0
    evaluation_threshold: float = 0.9

    # Store the final textual result for reference
    result: str = ""
    is_completed: bool = False


class BaseTool:
    name: str
    description: str

    def invoke(self, arguments: Dict[str, Any]) -> str:
        """
        Example method for tool usage. You can override it with your own logic.
        """
        return "Tool invoked with no actual logic"


class LatentPlanner:
    """
    A Planner that uses 'latent space' iterative reasoning for each step.
    1) At execution time, each step's latent_vector is updated through multiple recurrent iterations
       (similar to the 'recurrent depth' concept in the cited paper).
    2) If any required info is missing, it uses a 'human in the loop' approach to request that info
       before continuing the iteration.
    3) After the final iteration, produce an output (possibly JSON or any text) describing the action or result.
    """

    logger = get_logger("LatentPlanner")

    # A default prompt to show how we might incorporate the final LLM call
    DEFAULT_EXECUTE_PROMPT = """
[Latent Reasoning Planner]

We have a root task: {root_task}

Current step: {step_name}
Step description: {step_desc}

Required info: {step_info_required}
Context so far:
{context}

Latent vector state (not directly shown to user):
{latent_vector_snapshot}

Generate a final JSON describing the result or tool call.
For example:
```json
{{
  "use_tool": false,
  "response": "some final text result"
}}
"""

    def __init__(
        self,
        model_name: str,
        latent_dim: int = 64,
        max_rec_iter: int = 4,
        allow_human_in_the_loop: bool = True,
        log_level: str = "INFO",
    ):
        """
        :param model_name: The LLM to call
        :param latent_dim: dimension of the hidden vector for each step
        :param max_rec_iter: maximum recurrent iteration for each step
        :param allow_human_in_the_loop: if True, will pause to request user input if info is missing
        """
        self.model_name = model_name
        self.latent_dim = latent_dim
        self.max_rec_iter = max_rec_iter
        self.allow_human_in_the_loop = allow_human_in_the_loop

        self.steps: List[LatentTaskStep] = []
        self.completed_steps: List[LatentTaskStep] = []
        self.logger.setLevel(log_level)

        self.context_manager = ContextManager()  # store overall context
        self.execute_prompt = self.DEFAULT_EXECUTE_PROMPT


    def plan(
        self,
        root_task: str,
        steps_info: List[Dict[str, Any]],
    ) -> List[LatentTaskStep]:
        """
        Define a sequence of steps (LatentTaskStep) for the given root_task.
        steps_info: a list of dict with keys:
            - name (str)
            - description (str)
            - use_tool (bool)
            - tool_name (str)   [optional]
            - info_required (List[str]) [optional]
        """
        self.logger.info(f"Planning for root_task: {root_task}")
        self.steps.clear()

        for s in steps_info:
            step = LatentTaskStep(
                name=s.get("name", "Unknown"),
                description=s.get("description", "No desc"),
                use_tool=s.get("use_tool", False),
                tool_name=s.get("tool_name", None),
                info_required=s.get("info_required", []),
                max_attempts=s.get("max_attempts", 2),
                evaluation_threshold=s.get("evaluation_threshold", 0.9),
            )
            self.steps.append(step)

        return self.steps


    def execute_plan(
        self,
        root_task: str,
        evaluators_enabled: bool = False,
        evaluators: Dict[str, BaseEvaluator] = None,
        background: str = "",
        tools: Dict[str, BaseTool] = None,
    ):
        """
        Executes the plan step by step. For each step:
        1) Perform latent space recurrent iteration
        2) Check if we need human input if some required info is not found
        3) Produce a final LLM-based output
        4) Evaluate (optional), and decide whether to continue or retry
        """
        if evaluators is None:
            evaluators = {}

        for step_idx, step in enumerate(self.steps, start=1):
            step.current_attempts = 0
            while step.current_attempts < step.max_attempts and not step.is_completed:
                step.current_attempts += 1
                self.logger.info(
                    f"Executing Step {step_idx}/{len(self.steps)}: {step.name} - attempt {step.current_attempts}"
                )

                # 1) Initialize latent vector if needed
                if step.latent_vector is None:
                    step.latent_vector = self._init_latent_vector()

                # 2) Recurrent iteration in latent space
                iteration_success = self._recurrent_iteration(step, root_task)

                if not iteration_success:
                    self.logger.warning(
                        f"Step {step.name}: iteration incomplete, missing info or user refused to provide. Break."
                    )
                    break

                # 3) Finalize result (call LLM to produce final JSON output)
                llm_output = self._finalize_step_output(step, root_task, background)

                # 4) Parse the output, optionally call tool
                final_text, used_tool = self._parse_and_maybe_invoke_tool(
                    llm_output, step, tools
                )

                # store the result in step
                step.result = final_text

                # 5) Evaluate (if enabled)
                pass_eval = True
                if evaluators_enabled:
                    eval_score = self._evaluate_step(step, final_text, evaluators)
                    self.logger.info(f"Eval score for step {step.name}: {eval_score:.3f}")
                    pass_eval = eval_score >= step.evaluation_threshold

                if pass_eval:
                    self.logger.info(f"Step {step.name} is completed.")
                    step.is_completed = True
                    self.completed_steps.append(step)
                    # add context
                    self.context_manager.add_context(f"Step {step.name}", final_text)
                else:
                    self.logger.warning(
                        f"Step {step.name} not passing threshold, attempt {step.current_attempts} ended."
                    )
                    if step.current_attempts >= step.max_attempts:
                        self.logger.error(f"Step {step.name} failed after max attempts.")
                # end while

        self.logger.info("All steps executed or exhausted attempts.")
        return self.completed_steps


    def _init_latent_vector(self) -> np.ndarray:
        """
        Randomly initialize latent vector (like the random init in the paper).
        """
        return np.random.randn(self.latent_dim).astype(np.float32)


    def _recurrent_iteration(self, step: LatentTaskStep, root_task: str) -> bool:
        """
        Perform multiple updates on the latent_vector, simulating 'recurrent depth' from the paper.

        Return False if we couldn't proceed due to missing info or user refused to give info.
        """

        # For demonstration, we do a fixed number of recurrences: self.max_rec_iter
        # In the referenced paper, the number of iterations can also be dynamically chosen
        # or we can implement early stopping if the update is small enough, etc.
        for i in range(self.max_rec_iter):
            # 1) Check if we have missing info
            missing_keys = self._check_missing_info(step)
            if missing_keys:
                # ask user
                if not self.allow_human_in_the_loop:
                    self.logger.warning("Missing info but human_in_the_loop is disabled.")
                    return False
                # Try to gather from user
                user_provided = self._human_in_the_loop(missing_keys, step)
                if not user_provided:
                    self.logger.warning("User refused or no info provided. Abort step.")
                    return False

            # 2) "Update" the latent_vector. For demonstration, do a random small perturbation + tanh
            gradient = np.random.normal(loc=0, scale=0.01, size=(self.latent_dim,)).astype(
                np.float32
            )
            step.latent_vector = np.tanh(step.latent_vector + gradient)

            # 3) optional: we could do early stopping if gradient < threshold, etc.
            # e.g., if np.linalg.norm(gradient)<1e-4: break

        return True


    def _check_missing_info(self, step: LatentTaskStep) -> List[str]:
        """
        Check step.info_required against the current context to see if something is missing.

        Return a list of missing info keys.
        We treat the planner's context_manager as the global store of known info.
        """
        missing = []
        for key in step.info_required:
            if key not in self.context_manager.context:
                missing.append(key)
        return missing


    def _human_in_the_loop(self, missing_keys: List[str], step: list[LatentTaskStep]) -> bool:
        """
        Request info from the user for each missing key.
        Return True if user provided all info, False if user declined or partial.
        """
        self.logger.info(f"Step {step.name} needs user input for keys: {missing_keys}")
        for key in missing_keys:
            user_val = input(
                f"[Human needed] Please provide value for '{key}' (or type 'cancel' to abort): "
            )
            if user_val.strip().lower() == "cancel":
                return False
            # store in context
            self.context_manager.add_context(key, user_val)
        return True


    def _finalize_step_output(
        self, step: LatentTaskStep, root_task: str, background: str
    ) -> str:
        """
        After the latent space iteration is done, call the LLM to produce a final output (e.g. JSON).
        We pass some snapshot of the latent_vector to the prompt if desired.
        """
        # We'll show the first 5 dims for demonstration
        snapshot = (
            step.latent_vector[:5].round(3).tolist()
            if step.latent_vector is not None
            else []
        )
        step_info_required = step.info_required or []

        prompt = self.execute_prompt.format(
            root_task=root_task,
            step_name=step.name,
            step_desc=step.description,
            step_info_required=step_info_required,
            context=self.context_manager.context_to_str(),
            latent_vector_snapshot=snapshot,
        )

        llm = ModelRegistry.get_model(self.model_name)
        llm_response = llm.process(prompt)

        return llm_response


    def _parse_and_maybe_invoke_tool(
        self, llm_output: str, step: LatentTaskStep, tools: Dict[str, BaseTool]
    ) -> Tuple[str, bool]:
        """
        Parse the LLM's output as JSON, see if we need to call a tool.
        Return (final_text, used_tool: bool).
        """
        used_tool = False
        final_text = ""
        # try parse as JSON
        cleaned = llm_output.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
            if data.get("use_tool", False) and step.use_tool and step.tool_name and tools:
                # call tool
                tool_instance = tools.get(step.tool_name)
                if tool_instance:
                    arguments = data.get("tool_arguments", {})
                    tool_resp = tool_instance.invoke(arguments)
                    final_text = f"[Tool invoked] {tool_resp}"
                    used_tool = True
                else:
                    final_text = "[Error] Tool requested but not found in 'tools'"
            else:
                final_text = data.get("response", "[No response found]")
        except:
            final_text = "[JSON parse error or invalid format]"
        return final_text, used_tool


    def _evaluate_step(
        self, step: LatentTaskStep, result_text: str, evaluators: Dict[str, BaseEvaluator]
    ) -> float:
        """
        If we have an evaluator for the step name or 'default', use it to get a 0..1 score.
        """
        if step.name in evaluators:
            evaluator = evaluators[step.name]
        else:
            evaluator = evaluators.get("default", None)

        if evaluator is None:
            self.logger.info("No evaluator found, default pass with 1.0")
            return 1.0

        # A minimal signature
        eval_result = evaluator.evaluate(
            root_task="",  # not used here
            node_task=step.description,
            result=result_text,
            background="",
            context=self.context_manager,
        )
        # assume the paper's approach: 0..40 mapped to 0..1
        numeric_score = float(eval_result.score) / 40.0
        return numeric_score
