from __future__ import annotations
from dataclasses import field
from typing import List, Optional, Any, Dict
from pydantic import BaseModel
from agent_core.evaluators.entities.evaluator_result import EvaluatorResult


class Step(BaseModel):

    name: str
    description: str
    prompt: Optional[str] = ""
    result: Optional[str] = ""
    use_tool: Optional[bool] = False
    tool_name: Optional[str] = None
    tool_args: Optional[Any] = None
    category: Optional[str] = "default"
    retries: Optional[List[Step]] = field(default_factory=list)
    evaluator_result: Optional[EvaluatorResult] = None
    action: str = "next"
    is_success: bool = True
    plan_name: int = 1

    def add_retry(self, step: Step):
        if self.retries is None:
            self.retries = list()
        self.retries.append(step)

    def enrich_success_step(self, plan_name):
        self.action = "next"
        self.is_success = True
        self.plan_name = plan_name

    def enrich_failure_step(self, action, plan_name):
        self.action = action
        self.is_success = False
        self.plan_name = plan_name

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
            "result": self.result
        }


class TracePlan(BaseModel):

    plan: List[Step]
    adjustment: Optional[Any] = None


class Steps(BaseModel):

    steps: List[Step] = field(default_factory=list)
    summary: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    trace_steps: List[Step] = field(default_factory=list)
    trace_plan: Dict[int, TracePlan] = field(default_factory=dict)

    def __str__(self):
        return self.execution_history_to_str()

    def add_success_step(self, step: Step):
        step.enrich_success_step(len(self.trace_plan))
        trace_plan = self.trace_plan[step.plan_name]
        final_step = trace_plan.plan[-1]
        if final_step.name == step.name:
            step.action = "end"
        self.trace_steps.append(step)
        self.steps.append(step)

    def add_failure_step(self, step: Step):
        step.enrich_failure_step("failure", len(self.trace_plan))
        self.trace_steps.append(step)

    def add_retry_step(self, step: Step):
        step.enrich_failure_step("retry", len(self.trace_plan))
        self.trace_steps.append(step)

    def adjust_plan(self, action, plan: List[Step], adjustment):
        index = len(self.trace_plan) + 1
        trace_plan = TracePlan(plan=plan, adjustment=adjustment)
        self.trace_plan[index] = trace_plan
        if self.trace_steps and len(self.trace_steps) > 0:
            final_step = self.trace_steps[-1]
            final_step.action = action

    def add_plan(self, plan: List[Step]):
        index = len(self.trace_plan) + 1
        trace_plan = TracePlan(plan=plan)
        self.trace_plan[index] = trace_plan

    def get_info(self):
        return [step.to_dict() for step in self.steps]

    def to_dict(self):
        # Convert the Steps instance to a dictionary
        return {"steps": self.get_info}

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
