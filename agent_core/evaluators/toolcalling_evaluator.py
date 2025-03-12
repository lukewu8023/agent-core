# evaluators/toolcalling_evaluator.py

import json
from typing import Optional

from .base_evaluator import BaseEvaluator
from .entities.evaluator_result import EvaluatorResult
from agent_core.utils.context_manager import ContextManager


class ToolCallingEvaluator(BaseEvaluator):
    """
    A simple evaluator that checks only whether the tool calling signature
    in `response` is valid or not. If valid -> score=1, else 0.
    Produces a JSON-based evaluation result.
    """

    def default_prompt(self):
        """
        We do not actually call the LLM for function calling validation,
        so we won't rely on a prompt. Just return an empty string.
        """
        return ""

    def evaluate(
        self,
        root_task: str,
        request: str,
        response: str,
        background: str,
        context_manager: ContextManager,
    ) -> EvaluatorResult:
        """
        1) Parse the 'response' to see if there's a valid function (tool) call structure.
        2) If 'use_tool' is True and 'tool_arguments' is a dict => valid => score=1
           Otherwise => score=0
        3) Return a JSON structure with the same shape as the other evaluators.
        """
        # Initialize defaults
        overall_score = 0
        justification = ""

        try:
            # Attempt to parse the 'response' (which should be normal text, but we expect JSON or something).
            # For safety, treat 'response' as if it might contain JSON. If we cannot parse it, that is invalid.
            data = None
            cleaned = response.strip()
            # in practice we might accept partial JSON. But let's do a quick parse:
            data = json.loads(cleaned)
            # Minimal check:
            if isinstance(data, dict):
                if data.get("use_tool") is True and isinstance(
                    data.get("tool_arguments"), dict
                ):
                    # Valid
                    overall_score = 40.0
                    justification = (
                        "Function calling signature and parameters appear valid."
                    )
                else:
                    justification = "Function calling signature is invalid or missing 'use_tool'/'tool_arguments'."
            else:
                justification = "No valid JSON object found in response."
        except Exception:
            justification = "Failed to parse JSON in the function-calling response."

        # Build standard JSON structure
        # We'll have only one 'point': 'function_call_validity'
        points = [
            {
                "criterion": "function_call_validity",
                "score": overall_score,
                "justification": justification,
            }
        ]

        total_score = overall_score  # since we have only one point
        recommendation = "Accept Output" if total_score == 40.0 else "Rerun Subtask"
        rerun_suggestion = (
            ""
            if recommendation == "Accept Output"
            else "Invalid function signature or arguments, please regenerate."
        )

        # We'll interpret the final numeric_score as overall_score / 40.0 => 1.0 or 0.0
        numeric_score = float(overall_score) / 40.0  # 0 or 1

        # The "details" field can carry the raw JSON
        details_json = {
            "points": points,
            "improvement_suggestion": rerun_suggestion,
        }

        # Create an EvaluatorResult object
        return EvaluatorResult(
            name=self.name(),
            decision=recommendation,
            score=numeric_score,  # normalized if you wish. Here it's either 0.0 or 1.0
            suggestion=rerun_suggestion,
            details=details_json,
            prompt=self.prompt
        )
