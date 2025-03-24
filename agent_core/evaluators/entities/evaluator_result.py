from pydantic import BaseModel, Field


class EvaluatorResult(BaseModel):
    """
    decision: "Accept Output" or "Rerun Subtask"
    score: a numeric measure of quality
    suggestion: suggestion to improve
    details: optional extra data, e.g., breakdown of scores
    """

    name: str = Field("generic")
    decision: str = Field("Accept Output")
    score: float = Field(1.0)
    evaluation_threshold: float = Field(0.9)
    suggestion: str = Field("")
    details: str | list | dict = Field("")
    prompt: str = Field("")

    def to_info(self) -> dict:
        return {
            "name": self.name,
            "decision": self.decision,
            "score": self.score,
            "details": self.details,
        }

    def to_log(self) -> str:
        return f"""Evaluator Threshold: {self.evaluation_threshold}, Score: {self.score}, Decision: {self.decision}, Suggestion: {self.suggestion}"""
