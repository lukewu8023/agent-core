# evaluators/toolcalling_evaluator.py

from .base_evaluator import BaseEvaluator, parse_scored_evaluation_response
from .entities.evaluator_result import EvaluatorResult
from agent_core.utils.context_manager import ContextManager


EVALUATOR_PROMPT = """
You are an expert evaluator of AI-generated tool calls. Evaluate the provided tool call based on the following criteria:
- Each criterion is scored on a scale of 1 to 5 (1 = very poor, 5 = excellent).
- For each criterion, provide a short justification.

1. **Correct Tool Selection** (Score 1-5): The agent invoked the correct tool for the given task.
2. **Parameter Accuracy** (Score 1-5): The tool call contains the correct parameters with expected values.
3. **Parameter Completeness** (Score 1-5): All required parameters are present; none are missing.
4. **Parameter Relevance** (Score 1-5): The provided parameters are meaningful and necessary for the tool’s intended function.
5. **Format Correctness** (Score 1-5): The tool call follows the expected JSON structure and data types.
6. **Contextual Consistency** (Score 1-5): The tool call aligns with prior steps and expected behavior within the task flow.
7. **Execution Readiness** (Score 1-5): The tool call is directly executable without requiring additional corrections or assumptions.
8. **Error Handling & Robustness** (Score 1-5): The tool call accounts for potential edge cases (e.g., missing data, boundary values).

**Background**
{background}

**Context**
{context}

**Description of Ultimate Task Goal**
{root_task}

**Description of Current Step**
{request}

**Output of current step**
{response}

Evaluation of Current Step:
**Return Format**:
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

**IMPORTANT**:
Return the result strictly in a valid JSON object, with no extra text.
If a correction is needed, provide it in "improvement_suggestion" (leave empty if all scores are 5).
"""


class ToolCallingEvaluator(BaseEvaluator):
    """
    A simple evaluator that checks only whether the tool calling signature
    in `response` is valid or not. If valid -> score=1, else 0.
    Produces a JSON-based evaluation result.
    """

    def default_prompt(self):
        return EVALUATOR_PROMPT

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
        prompt_text = self.prompt.format(
            root_task=root_task,
            request=request,
            response=response,
            background=background,
            context=context_manager.context_to_str(),
        )
        evaluation_response = self._model.process(prompt_text)

        # parse the JSON
        decision, score, suggestion, details = parse_scored_evaluation_response(
            self.evaluation_threshold,
            evaluation_response
        )

        return EvaluatorResult(
            name=self.name,
            decision=decision,
            score=score,
            suggestion=suggestion,
            details=details,
            prompt=prompt_text,
        )