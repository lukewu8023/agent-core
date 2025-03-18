# evaluators/base_evaluator.py

from abc import abstractmethod
from typing import Optional
import json
from agent_core.agent_basic import AgentBasic
from agent_core.evaluators.entities.evaluator_result import EvaluatorResult
from agent_core.utils.context_manager import ContextManager


def parse_scored_evaluation_response(evaluation_threshold, evaluation_response: str):
    """
    Attempt to parse the JSON. If invalid, treat as "Rerun Subtask" with 0 score.
    """
    try:
        cleaned = (
            evaluation_response.replace("```json", "").replace("```", "").strip()
        )
        data = json.loads(cleaned)
        points = data.get("points", [])

        """Extract scores from JSON and calculate the total score."""
        scores = [point["score"] for point in points]
        total_score = sum(scores)

        # Check if any criterion scored below 3
        any_low_scores = any(score < 3 for score in scores)

        # Convert total_score from 0..40 to 0..1 scale for the node's final check
        numeric_score = float(total_score) / 40.0

        # Final decision logic
        if numeric_score > evaluation_threshold and not any_low_scores:
            recommendation = "Accept Output"
        else:
            recommendation = "Rerun Subtask"

        improvement_suggestion = data.get("improvement_suggestion", "")
        # We will keep the entire JSON in details for reference
        details = data

        return recommendation, numeric_score, improvement_suggestion, details
    except Exception as e:
        # If parse fails, fallback
        details = {"parse_error": str(e), "raw_response": evaluation_response}
        return "Rerun Subtask", 0, "Rerun this step.", details


class BaseEvaluator(AgentBasic):
    """
    A base class for all evaluator. Every evaluator must implement `evaluator()`.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        log_level: Optional[str] = None,
        evaluation_threshold: Optional[float] = 0.8,
        max_attempt: Optional[int] = 3,
    ):
        """
        Pass in the agent's model instance so we can call model.process(...) for evaluation prompts.
        Optionally specify log_level for debug or other logs.
        'prompt' can override the default prompt template.
        """
        super().__init__(self.__class__.__name__, model_name, log_level)
        self.evaluation_threshold = evaluation_threshold
        self.prompt = self.default_prompt()
        self.max_attempt = max_attempt
        self.name = self.__class__.__name__

    @abstractmethod
    def default_prompt(self):
        pass

    @abstractmethod
    def evaluate(
        self,
        root_task: str,
        request: str,
        response: str,
        background: str,
        context_manager: ContextManager,
    ) -> EvaluatorResult:
        """
        Perform evaluator on the given request and response.
        """
        pass
