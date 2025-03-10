from pydantic import BaseModel, Field


class EvaluatorResult(BaseModel):

    """
    decision: "Accept Output" or "Rerun Subtask"
    score: a numeric measure of quality
    suggestion: suggestion to improve
    details: optional extra data, e.g., breakdown of scores
    """
    decision: str = Field("")
    score: float = Field(0.0)
    suggestion: str = Field("")
    details: str | list | dict = Field("")
    prompt: str = Field("")

    def to_log(self):
        return f"""Evaluator Decision: {self.decision}, score: {self.score}, suggestion: {self.suggestion}"""