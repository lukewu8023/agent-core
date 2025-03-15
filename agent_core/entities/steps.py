from __future__ import annotations
from dataclasses import field
from typing import List, Optional
from pydantic import BaseModel
from agent_core.evaluators.entities.evaluator_result import EvaluatorResult


class Step(BaseModel):

    name: str
    description: str
    prompt: Optional[str] = ""
    result: Optional[str] = ""
    use_tool: Optional[bool] = False
    tool_name: Optional[str] = None
    category: Optional[str] = "default"
    retries: Optional[List[Step]] = field(default_factory=list)
    evaluator_result: Optional[EvaluatorResult] = None

    def add_retry(self, step: Step):
        if self.retries is None:
            self.retries = list()
        self.retries.append(step)

    def add_evaluator_result(self, evaluator_result: EvaluatorResult):
        self.evaluator_result = evaluator_result

    def to_success_info(self) -> str:
        return f"""Step : {self.name} Description: {self.description} Result: {self.result}\n"""

    def get_info(self) -> dict:
        info = self.to_dict()
        info["result"] = self.result
        info["evaluator_result"] = self.evaluator_result.to_info()
        info["retries"] = [step.evaluator_result.to_info() for step in self.retries]
        return info

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

    steps: List[Step] = field(default_factory=list)
    token: int = 0

    def __str__(self):
        return self.execution_history_to_str()

    def add_step(self, step: Step):
        self.steps.append(step)

    def get_info(self):
        return [step.get_info() for step in self.steps]

    def to_dict(self):
        # Convert the Steps instance to a dictionary
        return {"steps": [step.to_dict() for step in self.steps]}

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
