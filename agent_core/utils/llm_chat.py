import re
import json
from typing import Optional
from agent_core.agent_basic import AgentBasic


def _parse_section(response_text: str, label: str) -> str:
    """
    Extract text after the given label, e.g. "Summary:" or "Suggestions:".
    Returns the found text or a default string.
    Example: label="summary" => look for "Summary:"
    """
    pattern = rf"{label}:\s*(.*)"
    match = re.search(pattern, response_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return f"No {label} found."


def _parse_rating(response_text: str) -> int:
    """
    Find 'Rating: X' in the text. Return X as an integer. Default to 1 if not found.
    """
    match = re.search(r"Rating:\s*(\d+)", response_text, re.IGNORECASE)
    if match:
        return min(10, max(1, int(match.group(1))))  # clamp rating to [1..10]
    return 1


class LLMChat(AgentBasic):

    DEFAULT_EVALUATE_TEXT_PROMPT = """\
    You are a critical evaluator. Below is some text to evaluate:
    Text:
    {input_text}
    Criteria:
    {criteria}
    Instructions:
    1) Provide a short explanation of how well the text meets the criteria.
    2) Then on a new line, output "Rating: X" where X is an integer in [1..10].
    3) Optionally provide suggestions.
    Example:
    Summary: This text partially meets the criteria but could be more clear.
    Rating: 7
    Suggestions: Make it clearer how the data is processed.
    Now, produce your evaluation:
    """

    def __init__(self, model_name: str = None, log_level: str = None):
        """
        If model_name is None, use the default model from Config.
        This class can be used to do both summarization and text-critique.
        """
        super().__init__(self.__class__.__name__, model_name, log_level)
        self._evaluate_text_prompt = self.DEFAULT_EVALUATE_TEXT_PROMPT

    @property
    def evaluate_text_prompt(self) -> str:
        """Prompt used for evaluating text."""
        return self._evaluate_text_prompt

    @evaluate_text_prompt.setter
    def evaluate_text_prompt(self, value: str):
        self._evaluate_text_prompt = value

    def process(self, request: str) -> str:
        response = self._model.process(request)
        self.logger.debug(f"Response: {response}")
        return response.strip()

    def evaluate_text(
        self, input_text: str, criteria: str = "", rating_threshold: int = 8
    ) -> dict:
        """
        Critique/evaluate the text using the DEFAULT_EVALUATE_TEXT_PROMPT.
        Returns a dict with keys:
          - 'decision': "Pass" or "Fail" based on rating_threshold
          - 'rating': the integer rating from 1..10
          - 'summary': the extracted text around "Summary:"
          - 'suggestions': text after "Suggestions:" if found
          - 'raw_response': the entire LLM response
        Example usage:
            result = llm.critic_text("some plan", "Should talk about DB usage", 8)
            if result["decision"] == "Fail":
                # handle test fail logic
        """
        prompt = self.DEFAULT_EVALUATE_TEXT_PROMPT.format(
            input_text=input_text,
            criteria=criteria,
        )
        response = self._model.process(prompt)
        self.logger.debug(f"Evaluate raw response: {response}")
        # Parse rating from 1..10
        rating_val = _parse_rating(response)
        decision = "Pass" if rating_val >= rating_threshold else "Fail"
        # Optionally parse 'Summary:' and 'Suggestions:'
        summary_str = _parse_section(response, "summary")
        suggestions_str = _parse_section(response, "suggestions")
        return {
            "decision": decision,
            "rating": rating_val,
            "summary": summary_str,
            "suggestions": suggestions_str,
            "raw_response": response,
        }

    def parse_llm_response(self, llm_response: str) -> Optional[str]:
        try:
            cleaned = llm_response.replace("```json", "").replace("```", "").strip()
            llm_response_json = json.loads(cleaned)
            return llm_response_json
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            return None
