# evaluators/coding_evaluator.py

import json
from typing import Optional, List
from .base_evaluator import BaseEvaluator
from .entities.evaluator_result import EvaluatorResult

reject_code: str = "Reject Code"


DEFAULT_PROMPT = """\
You are an expert code reviewer with extensive experience in software engineering and multiple programming languages.
Your task is to evaluate the provided AI-generated code based on the following comprehensive criteria. Note: The provided code may be a complete implementation or just a code fragment. If a particular criterion is not applicable, please mark it as "N/A" and exclude it from the overall scoring.

1. **Requirements Coverage** (Score 1-5 or N/A): Evaluate to what extent the code addresses and fulfills the specified requirements. For a code fragment, consider whether it aligns with or partially implements the requirements.
2. **Correctness** (Score 1-5 or N/A): Assess whether the code implements correct logic and produces the expected outputs based on the provided requirements.
3. **Code Style and Conventions** (Score 1-5 or N/A): Determine if the code follows standard coding conventions and best practices.
4. **Readability and Documentation** (Score 1-5 or N/A): Evaluate how clear and well-documented the code is.
5. **Efficiency and Performance** (Score 1-5 or N/A): Assess whether the code is optimized for performance and resource usage.
6. **Maintainability and Scalability** (Score 1-5 or N/A): Evaluate if the code is structured in a way that makes it easy to maintain and extend, considering potential integration of the fragment.
7. **Security and Robustness** (Score 1-5 or N/A): Assess if the code incorporates proper error handling, security best practices, and robustness against edge cases.
8. **Testability** (Score 1-5 or N/A): Determine if the code is designed to be easily testable or includes testing provisions.

For each criterion, provide:

- **Score (1-5) or N/A**
- **Justification:** A brief explanation for your score or reason for marking it N/A.

At the end, return a JSON object with the following fields, ensuring it is a valid JSON string:

- `decision`: "Accept Code" or "Reject Code"
- `total_applicable_score`: The total score of applicable criteria.
- `average_score`: The average score of applicable criteria.
- `scores`: An array of objects with `criterion`, `score`, and `justification` for each criterion.
- `improvement_suggestions`: A string with improvement suggestions for each criterion that scored below 3.

Ensure the output is only the JSON string, with no additional characters, headers, or formatting.

**Context**
{context}

**Description of ultimate task goal:**
{root_task}

**Request:**
{request}

**Response:**
{response}

**Evaluation (JSON Format):**
"""


def generate_improvement_suggestions(scores: List[tuple]) -> str:
    """
    Generate improvement suggestions based on the scores. For each applicable criterion with a score below 3,
    provide a specific suggestion.
    """
    suggestion_dict = {
        "Requirements Coverage": "Ensure the code fully addresses and implements the requirements.",
        "Correctness": "Review the logic and test the code to verify its correctness.",
        "Code Style and Conventions": "Follow standard coding practices and style guidelines.",
        "Readability and Documentation": "Improve code organization and add meaningful comments and documentation.",
        "Efficiency and Performance": "Optimize algorithms and resource usage for better performance.",
        "Maintainability and Scalability": "Refactor the code to enhance maintainability and facilitate future extensions.",
        "Security and Robustness": "Implement proper error handling and security measures to cover edge cases.",
        "Testability": "Design the code structure to facilitate unit testing and overall verification.",
    }

    suggestions = []
    for criterion, score_value in scores:
        if score_value < 3:
            suggestion = suggestion_dict.get(
                criterion, f"Consider improving {criterion}."
            )
            suggestions.append(f"- {criterion}: {suggestion}")

    if not suggestions:
        suggestions.append(
            "Review the overall implementation to better meet the requirements and quality standards."
        )

    return "\n".join(suggestions)


class CodingEvaluator(BaseEvaluator):

    def __init__(
        self,
        model_name: Optional[str] = None,
        log_level: Optional[str] = None,
        evaluation_threshold: Optional[float] = 0.8,
    ):
        super().__init__(model_name, log_level, evaluation_threshold)

    def evaluate(
        self, root_task, request, response, background, context_manager
    ) -> EvaluatorResult:
        """
        Evaluate the provided request and generated code response.
        """
        prompt_text = self.prompt.format(
            root_task=root_task,
            request=request,
            response=response,
            background=background,
            context=context_manager.context_to_str(),
        )

        try:
            evaluation_response = self._model.process(prompt_text)
        except Exception as e:
            self.logger.error("Error during model evaluation: %s", e)
            return EvaluatorResult(
                decision=reject_code,
                score=0,
                details={
                    "score_breakdown": [],
                    "raw_evaluation": "",
                    "improvement_suggestions": "Model evaluation failed. Please try again.",
                },
            )

        result = self.parse_scored_evaluation_response(evaluation_response)

        if isinstance(result, EvaluatorResult):
            # Already an EvaluatorResult, just return it
            return result
        else:
            # Legacy tuple return format
            decision, total_score, scores = result

            details = {
                "score_breakdown": scores,
                "raw_evaluation": evaluation_response,
                "total_applicable_score": total_score,
            }

            if decision == reject_code:
                improvement_suggestions = generate_improvement_suggestions(scores)
                details["improvement_suggestions"] = improvement_suggestions
            else:
                details["improvement_suggestions"] = ""

            return EvaluatorResult(decision, total_score, details)

    def default_prompt(self):
        return DEFAULT_PROMPT

    def parse_scored_evaluation_response(self, evaluation_response: str):
        """
        Parse the evaluation text to extract scores for each criterion, and calculate the total and average scores.
        For criteria marked as N/A, they are excluded from the score calculation.
        """
        try:
            cleaned_evaluation_response = (
                evaluation_response.replace("```json", "").replace("```", "").strip()
            )
            evaluation_data = json.loads(cleaned_evaluation_response)
        except json.JSONDecodeError:
            self.logger.error("Unable to parse the model's evaluation response.")
            return reject_code, 0, []

        scores = []
        total_score = 0
        applicable_criteria_count = 0

        for item in evaluation_data["scores"]:
            score_value = item["score"]
            if score_value != "N/A":
                scores.append((item["criterion"], score_value))
                total_score += score_value
                applicable_criteria_count += 1

        if applicable_criteria_count == 0:
            self.logger.warning("No valid criteria were parsed for evaluation.")
            return reject_code, 0, scores

        # Check if any criterion scored below 3
        any_low_scores = any(score < 3 for _, score in scores)
        average_score = total_score / applicable_criteria_count

        decision = (
            "Accept Code"
            if average_score >= (5 * self.evaluation_threshold) and not any_low_scores
            else reject_code
        )

        return decision, total_score, scores
