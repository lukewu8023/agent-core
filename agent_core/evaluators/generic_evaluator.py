# evaluator/generic_evaluator.py

from typing import Optional
from .base_evaluator import BaseEvaluator, parse_scored_evaluation_response
from .entities.evaluator_result import EvaluatorResult

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

IMPORTANT: Return your result strictly in a valid JSON object and nothing else, with this structure:

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
- If output is 'incorrect tool arguments and unexpected result' when invoke the tool, provide the correcting suggestion and the structure evaluation output still (Score 0 for each criterion)
- If output is 'Invalid JSON format in response', provide the structure evaluation output still (Score 0 for each criterion)

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


class GenericEvaluator(BaseEvaluator):

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

    def default_prompt(self):
        return DEFAULT_PROMPT
