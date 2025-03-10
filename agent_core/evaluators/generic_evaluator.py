# evaluator/generic_evaluator.py

import re
import json
from typing import Optional
from .base_evaluator import BaseEvaluator
from .entities.evaluator_result import EvaluatorResult


class GenericEvaluator(BaseEvaluator):
    DEFAULT_PROMPT = """
You are an expert evaluator of AI-generated outputs. Evaluate the provided subtask output based on the following criteria:
- Each criterion is scored on a scale of 1 to 5 (1=very poor, 5=excellent). 
- For each criterion provide a short justification.

1. **Accuracy** (Score 1-5): The output fulfills the requirements of the subtask accurately.
2. **Completeness** (Score 1-5): The output addresses all aspects of the subtask.
3. **Relevance** (Score 1-5): The content is directly relevant to the subtask without extraneous information.
4. **Coherence and Clarity** (Score 1-5): The output is logically structured, clear, and easy to understand.
5. **Consistency** (Score 1-5): The output is consistent with previous subtasks and doesn't contradict itself.
6. **Following Instructions** (Score 1-5): The output adheres to any specific instructions or formats specified.
7. **Error Analysis** (Score 1-5): The output is free from factual, grammatical, and logical errors.
8. **Ethical Compliance** (Score 1-5): The content complies with ethical guidelines and policies.

At the end:
Based on the justifications of all criteria, provide a **improvement_suggestion** with any improvement suggestions to reach the full score (empty if full score - all scores are 5).

IMPORTANT: Return your result strictly in **valid JSON** and nothing else, with this structure:

{{
  "points": [
    {{
      "criterion": "<string>",
      "score": <integer>,
      "justification": "<string>"
    }},
    ...
  ],
  "improvement_suggestion": "<string>"
}}

Do not include any extra keys or text outside this JSON.

- If output is an incorrect and unexpected structure in response, provide the structure evaluation output still (Score 0 for each criterion)
- If output is 'incorrect tool arguments and unexpected result' when invoke the tool, provide the change suggestion and the structure evaluation output still (Score 0 for each criterion)

---

**Background**
{background}

**Context**
{context}

**Description of ultimate task goal:**
{root_task}

**Description of current Step:**
{request}

**Output of current step:**
{response}

**Evaluation of current step:**
"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        log_level: Optional[str] = None,
        evaluation_threshold: Optional[float] = 0.9,
    ):
        super().__init__(model_name, log_level, evaluation_threshold)

    def evaluate(
        self, root_task, request, response, background, context_manager
    ) -> EvaluatorResult:
        """
        1) Build the evaluation prompt
        2) Get the LLM's JSON-based evaluation
        3) Parse it
        """
        prompt_text = self.prompt.format(
            root_task=root_task,
            request=request,
            response=response,
            background=background,
            context=context_manager.context_to_str(),
        )
        evaluation_response = self._model.process(prompt_text)

        # parse the JSON
        decision, score, suggestion, details = self.parse_scored_evaluation_response(
            evaluation_response
        )

        return EvaluatorResult(decision=decision, score=score, suggestion=suggestion, details=details,
                               prompt=self.default_prompt())

    def default_prompt(self):
        return self.DEFAULT_PROMPT

    def parse_scored_evaluation_response(self, evaluation_response: str):
        """
        Attempt to parse the JSON. If invalid, treat as "Rerun Subtask" with 0 score.
        """
        try:
            data = json.loads(evaluation_response)
            points = data.get("points", [])

            """Extract scores from JSON and calculate the total score."""
            scores = [point["score"] for point in points]
            total_score = sum(scores)

            # Check if any criterion scored below 3
            any_low_scores = any(score < 3 for _, score in scores)

            # Convert total_score from 0..40 to 0..1 scale for the node's final check
            numeric_score = float(total_score) / 40.0

            # Final decision logic
            if numeric_score > self.evaluation_threshold and not any_low_scores:
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
            return ("Rerun Subtask", 0, "Rerun this step.", details)
