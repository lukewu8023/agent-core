from __future__ import annotations
from typing import List, Optional, Annotated, Any
from pydantic import BaseModel, Field
from agent_core.evaluators.entities.evaluator_result import EvaluatorResult


class Step(BaseModel):

    name: str
    description: str
    prompt: Optional[str] = Field(None)
    result: Optional[str] = Field(None)
    use_tool: Optional[bool] = Field(None)
    tool_name: Optional[str] = Field(None)
    category: Optional[str] = Field("default")
    retries: Optional[List[Step]] = Field(None)
    evaluator_result: Optional[EvaluatorResult] = Field(None)

    def add_retry(self, step: Step):
        if self.retries is None:
            self.retries = list()
            self.retries.append(step)

    def add_evaluator_result(self, evaluator_result: EvaluatorResult):
        self.evaluator_result = evaluator_result

    def to_success_context(self, idx):
        return f"""Step {idx}: {self.name} Description: {self.description} Result: {self.result}\n"""

    def to_dict(self):
        # Convert the object to a dictionary that can be serialized to JSON
        return {
            "name": self.name,
            "description": self.description,
            "use_tool": self.use_tool,
            "tool_name": self.tool_name,
            "category": self.category,
        }


class Steps(BaseModel):

    steps: List[Step]

    def __init__(self, /, **data: Any):
        super().__init__(**data)
        self.steps = []

    def __str__(self):
        return self.execution_history_to_str()

    def add_step(self, step: Step):
        self.steps.append(step)

    def to_dict(self):
        # Convert the Steps instance to a dictionary
        return {
            "steps": [step.to_dict() for step in self.steps]
        }

    def get_last_step_output(self):
        if self.steps is not None and len(self.steps) > 0:
            return self.steps[-1]
        else:
            return ""

    # Build a textual representation of the execution history
    def execution_history_to_str(self):
        history_lines = []
        for idx, step in enumerate(self.steps, 1):
            line = (
                f"Step {idx}: {step.name}\n"
                f"Description: {step.description}\n"
                f"Result: {step.result}\n"
            )
            history_lines.append(line)
        history_text = "\n".join(history_lines)
        return history_text

    def execution_history_to_responses(self):
        response_lines = []
        for step in self.steps:
            line = f"{step.result}\n"
            response_lines.append(line)

        responses_text = "".join(response_lines)
        if responses_text.endswith("\n"):
            responses_text = responses_text[:-1]

        return responses_text

