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
    suggestion: str = Field("")
    details: str | list | dict = Field("")
    prompt: str = Field("")

    def to_info(self) -> dict:
        return {
            "name": self.name,
            "decision": self.decision,
            "score": self.score,
            "details": self.details
        }

    def to_log(self) -> str:
        return f"""Evaluator Decision: {self.decision}, score: {self.score}, suggestion: {self.suggestion}"""
