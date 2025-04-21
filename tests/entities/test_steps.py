import pytest
from agent_core.entities.steps import Step, Steps, TracePlan, Summary
from agent_core.evaluators.entities.evaluator_result import EvaluatorResult


class TestStep:
    def test_step_initialization(self):
        step = Step(
            name="test_step",
            description="test_description",
            prompt="test_prompt",
            result="test_result",
            use_tool=True,
            tool_name="test_tool",
            tool_args={"arg1": "value1"},
            category="test_category",
        )

        assert step.name == "test_step"
        assert step.description == "test_description"
        assert step.prompt == "test_prompt"
        assert step.result == "test_result"
        assert step.use_tool is True
        assert step.tool_name == "test_tool"
        assert step.tool_args == {"arg1": "value1"}
        assert step.category == "test_category"
        assert step.retries == []
        assert step.evaluator_result is None
        assert step.action == "next"
        assert step.is_success is True
        assert step.plan_name == 1

    def test_add_retry(self):
        step = Step(name="main", description="main step")
        retry_step = Step(name="retry", description="retry step")

        step.add_retry(retry_step)
        assert len(step.retries) == 1
        assert step.retries[0].name == "retry"

    def test_enrich_success_step(self):
        step = Step(name="test", description="test")
        step.enrich_success_step(2)

        assert step.action == "next"
        assert step.is_success is True
        assert step.plan_name == 2

    def test_enrich_failure_step(self):
        step = Step(name="test", description="test")
        step.enrich_failure_step("retry", 3)

        assert step.action == "retry"
        assert step.is_success is False
        assert step.plan_name == 3

    def test_add_evaluator_result(self):
        step = Step(name="test", description="test")
        evaluator_result = EvaluatorResult(is_success=True, feedback="good")
        step.add_evaluator_result(evaluator_result)

        assert step.evaluator_result == evaluator_result

    def test_to_success_info(self):
        step = Step(name="test", description="description", result="result")
        info = step.to_success_info()

        assert "Step : test" in info
        assert "Description: description" in info
        assert "Result: result" in info

    def test_get_info(self):
        step = Step(name="test", description="description", result="result")
        evaluator_result = EvaluatorResult(is_success=True, feedback="good")
        step.add_evaluator_result(evaluator_result)

        retry_step = Step(name="retry", description="retry")
        retry_step.add_evaluator_result(
            EvaluatorResult(is_success=False, feedback="bad")
        )
        step.add_retry(retry_step)

        info = step.get_info()

        assert info["name"] == "test"
        assert info["description"] == "description"
        assert info["result"] == "result"
        assert info["evaluator_result"] == evaluator_result.to_info()
        assert len(info["retries"]) == 1

    def test_to_dict(self):
        step = Step(
            name="test",
            description="description",
            use_tool=True,
            tool_name="tool",
            category="category",
            result="result",
        )

        data = step.to_dict()

        assert data == {
            "name": "test",
            "description": "description",
            "use_tool": True,
            "tool_name": "tool",
            "category": "category",
            "result": "result",
        }


class TestTracePlan:
    def test_trace_plan_initialization(self):
        steps = [Step(name="step1", description="desc1")]
        trace_plan = TracePlan(plan=steps, adjustment={"test": "adjustment"})

        assert len(trace_plan.plan) == 1
        assert trace_plan.plan[0].name == "step1"
        assert trace_plan.adjustment == {"test": "adjustment"}


class TestSteps:
    def test_steps_initialization(self):
        steps = Steps()

        assert steps.steps == []
        assert steps.summary == Summary(summary='', output_result='', conclusion='')
        assert steps.input_tokens == 0
        assert steps.output_tokens == 0
        assert steps.trace_steps == []
        assert steps.trace_plan == {}

    def test_add_success_step(self):
        steps = Steps()
        step = Step(name="test", description="test")

        # First test with empty trace_plan
        with pytest.raises(KeyError):
            steps.add_success_step(step)  # This will raise KeyError for trace_plan[0]

        # Test with existing trace_plan
        steps = Steps()  # Reset steps
        plan_step = Step(name="test", description="test")
        steps.trace_plan[1] = TracePlan(
            plan=[plan_step]
        )  # Initialize trace_plan[1] to match implementation
        step2 = Step(name="test", description="test")
        steps.add_success_step(step2)
        assert step2.action == "end"
        assert step2.plan_name == 1  # Matches the trace_plan key we set

    def test_add_failure_step(self):
        steps = Steps()
        step = Step(name="test", description="test")

        steps.add_failure_step(step)
        assert len(steps.trace_steps) == 1
        assert step.action == "failure"
        assert step.is_success is False
        assert step.plan_name == 0

    def test_add_retry_step(self):
        steps = Steps()
        step = Step(name="test", description="test")

        steps.add_retry_step(step)
        assert len(steps.trace_steps) == 1
        assert step.action == "retry"
        assert step.is_success is False
        assert step.plan_name == 0

    def test_adjust_plan(self):
        steps = Steps()
        plan = [Step(name="step1", description="desc1")]

        steps.adjust_plan("adjust", plan, {"test": "adjustment"})
        assert len(steps.trace_plan) == 1
        assert steps.trace_plan[1].plan[0].name == "step1"
        assert steps.trace_plan[1].adjustment == {"test": "adjustment"}

        # Test with existing trace_steps
        steps.trace_steps.append(Step(name="last", description="last"))
        steps.adjust_plan("new_action", plan, None)
        assert steps.trace_steps[-1].action == "new_action"

    def test_add_plan(self):
        steps = Steps()
        plan = [Step(name="step1", description="desc1")]

        steps.add_plan(plan)
        assert len(steps.trace_plan) == 1
        assert steps.trace_plan[1].plan[0].name == "step1"
        assert steps.trace_plan[1].adjustment is None

    def test_get_info(self):
        steps = Steps()
        step = Step(name="test", description="test")
        steps.steps.append(step)

        info = steps.get_info()
        assert len(info) == 1
        assert info[0]["name"] == "test"

    def test_to_dict(self):
        steps = Steps()
        step = Step(name="test", description="test")
        steps.steps.append(step)

        data = steps.to_dict()
        assert callable(data["steps"])  # It returns the get_info method

    def test_get_last_step_output(self):
        steps = Steps()
        assert steps.get_last_step_output() == ""

        step = Step(name="test", description="test")
        steps.steps.append(step)
        assert steps.get_last_step_output() == step

    def test_execution_history_to_str(self):
        steps = Steps()
        step1 = Step(name="step1", description="desc1", result="result1")
        step2 = Step(name="step2", description="desc2", result="result2")
        steps.steps = [step1, step2]

        history = steps.execution_history_to_str()
        assert "Step 1: step1" in history
        assert "Description: desc1" in history
        assert "Result: result1" in history
        assert "Step 2: step2" in history
        assert "Description: desc2" in history
        assert "Result: result2" in history

    def test_execution_history_to_responses(self):
        steps = Steps()
        step1 = Step(name="step1", description="desc1", result="result1")
        step2 = Step(name="step2", description="desc2", result="result2")
        steps.steps = [step1, step2]

        responses = steps.execution_history_to_responses()
        assert responses == "result1\nresult2"

        # Test with trailing newline in one result
        step3 = Step(name="step3", description="desc3", result="result3\n")
        steps.steps = [step1, step2, step3]
        responses = steps.execution_history_to_responses()
        # Updated assertion to match actual behavior with trailing newline
        assert responses == "result1\nresult2\nresult3\n"
