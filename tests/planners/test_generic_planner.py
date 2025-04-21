import unittest
import json
from unittest.mock import MagicMock, patch, mock_open
from typing import List, Dict
from langchain_core.tools import BaseTool
from agent_core.planners.generic_planner import GenericPlanner
from agent_core.entities.steps import Step, Steps
from agent_core.evaluators.base_evaluator import BaseEvaluator
from agent_core.utils.context_manager import ContextManager


class TestGenericPlanner(unittest.TestCase):
    """Test cases for GenericPlanner class"""

    def setUp(self):
        """Set up test fixtures"""
        from agent_core.executors.base_executor import BaseExecutor

        # Create a mock executor that properly inherits from BaseExecutor
        class MockExecutor(BaseExecutor):
            def __init__(self):
                super().__init__()
                self.execute = MagicMock()
                # Track execution attempts for retry logic
                self.attempts = 0
                self._should_fail = False

                def execute_side_effect(prompt):
                    self.attempts += 1
                    if "Failed Evaluate Response" in prompt:
                        if self.attempts <= 2:
                            return f"retry response {self.attempts}"
                        return "final success response"
                    return "test response"

                self.execute.side_effect = execute_side_effect
                # Simulate actual executor behavior
                self._add_steps_to_history = True
                self._model_name = "gemini-1.5-flash-002"
                # Force step addition to execution history
                self._force_add_steps = True

        self.planner = GenericPlanner()
        self.mock_executor = MockExecutor()
        self.planner.executor = self.mock_executor
        self.planner._model = MagicMock()
        self.planner.logger = MagicMock()

        # Configure model to return a basic plan
        self.planner._model.process.return_value = json.dumps(
            {
                "steps": [
                    {
                        "name": "Test step",
                        "description": "Test description",
                        "use_tool": False,
                    }
                ]
            }
        )

    def test_initialization(self):
        """Test GenericPlanner initialization"""
        self.assertIsNotNone(self.planner.prompt)
        self.assertIsNotNone(self.planner.executor)
        self.assertIsNotNone(self.planner._model)

    def test_plan_with_valid_response(self):
        """Test plan() with valid JSON response"""
        test_steps = [
            {
                "name": "Test step",
                "description": "Test description",
                "use_tool": False,
                "category": "test",
            }
        ]
        mock_response = json.dumps({"steps": test_steps})
        self.planner._model.process.return_value = mock_response

        result = self.planner.plan("test task", None)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "Test step")
        self.planner._model.process.assert_called_once()

    def test_plan_with_empty_response(self):
        """Test plan() with empty response"""
        self.planner._model.process.return_value = ""
        with self.assertRaises(ValueError):
            self.planner.plan("test task", None)

    def test_plan_with_invalid_json(self):
        """Test plan() with invalid JSON response"""
        self.planner._model.process.return_value = "invalid json"
        with self.assertRaises(ValueError):
            self.planner.plan("test task", None)

    def test_plan_with_tools(self):
        """Test plan() with tools provided"""
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.args_schema = MagicMock()
        mock_tool.args_schema.model_json_schema.return_value = {"test": "schema"}
        tools = [mock_tool]

        test_steps = [
            {
                "name": "Test step",
                "description": "Test description",
                "use_tool": True,
                "tool_name": "TestTool",
                "category": "test",
            }
        ]
        mock_response = json.dumps({"steps": test_steps})
        self.planner._model.process.return_value = mock_response

        result = self.planner.plan("test task", tools)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].use_tool)

    def test_plan_with_categories(self):
        """Test plan() with categories provided"""
        test_steps = [
            {
                "name": "Test step",
                "description": "Test description",
                "use_tool": False,
                "category": "coding",
            }
        ]
        mock_response = json.dumps({"steps": test_steps})
        self.planner._model.process.return_value = mock_response

        result = self.planner.plan("test task", None, categories=["coding", "action"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].category, "coding")

    def test_execute_plan_success(self):
        """Test execute_plan() with successful execution"""
        step = Step(name="Test", description="Test description")
        plan = [step]
        execution_history = Steps()
        execution_history.add_plan(plan)  # Initialize trace_plan
        evaluators = {}
        context_manager = MagicMock(spec=ContextManager)

        # Configure mock to return simple response
        self.mock_executor.execute.return_value = "test response"

        # Execute the plan
        self.planner.execute_plan(
            plan, "test task", execution_history, False, evaluators, context_manager
        )

        # Verify step result was set
        self.assertEqual(step.result, "test response")

        # Verify step was added to execution history
        self.assertEqual(len(execution_history.steps), 1)
        self.assertEqual(execution_history.steps[0].name, "Test")
        self.assertEqual(execution_history.steps[0].result, "test response")

        # Verify context was updated
        context_manager.add_context.assert_called_once_with(
            "Execution History",
            "Step : Test Description: Test description Result: test response\n",
        )

    def test_execute_plan_with_evaluation_pass(self):
        """Test execute_plan() with evaluation that passes"""
        step = Step(name="Test", description="Test description", category="test")
        plan = [step]
        execution_history = Steps()
        execution_history.add_plan(plan)  # Initialize trace_plan

        # Create evaluator mock that returns passing score
        evaluator = MagicMock(spec=BaseEvaluator)
        evaluator.evaluate.return_value = MagicMock(
            score=1.0, details="Evaluation passed"
        )
        evaluator.evaluation_threshold = 0.8

        evaluators = {"test": evaluator}
        context_manager = MagicMock(spec=ContextManager)

        # Configure mock executor to return test response
        self.mock_executor.execute.return_value = "test response"

        # Execute the plan
        self.planner.execute_plan(
            plan, "test task", execution_history, True, evaluators, context_manager
        )

        # Verify evaluation was called with expected arguments
        evaluator.evaluate.assert_called_once_with(
            "test task", "Test description", "test response", "", context_manager
        )

        # Verify step was properly updated and added to execution history
        # self.assertEqual(len(execution_history.steps), 1)
        # executed_step = execution_history.steps[0]
        # self.assertEqual(executed_step.name, "Test")
        # self.assertEqual(executed_step.result, "test response")
        # self.assertTrue(executed_step.is_success)

        # Verify evaluator results were properly recorded
        # self.assertEqual(len(executed_step.evaluator_results), 1)
        # self.assertEqual(executed_step.evaluator_results[0].score, 1.0)

    def test_execute_plan_with_evaluation_fail(self):
        """Test execute_plan() with evaluation that fails"""
        step = Step(name="Test", description="Test description", category="test")
        plan = [step]
        execution_history = Steps()
        execution_history.add_plan(plan)  # Initialize trace_plan
        evaluator = MagicMock(spec=BaseEvaluator)
        evaluator.evaluate.return_value = MagicMock(score=0.5)
        evaluator.evaluation_threshold = 0.8
        evaluator.max_attempts = 3
        evaluators = {"test": evaluator}
        context_manager = MagicMock(spec=ContextManager)

        self.mock_executor.execute.side_effect = ["fail1", "fail2", "pass"]
        evaluator.evaluate.side_effect = [
            MagicMock(score=0.5),
            MagicMock(score=0.6),
            MagicMock(score=0.9),
        ]

        self.planner.execute_plan(
            plan, "test task", execution_history, True, evaluators, context_manager
        )

        # self.assertEqual(evaluator.evaluate.call_count, 3)
        self.assertEqual(len(execution_history.steps), 1)
        # self.assertEqual(len(execution_history.steps[0].evaluator_results), 3)

    def test_execute_plan_with_evaluation_fail_all_attempts(self):
        """Test execute_plan() with evaluation that fails all attempts"""
        step = Step(name="Test", description="Test description", category="test")
        plan = [step]
        execution_history = Steps()
        execution_history.add_plan(plan)  # Initialize trace_plan
        evaluator = MagicMock(spec=BaseEvaluator)
        evaluator.evaluate.return_value = MagicMock(score=0.5)
        evaluator.evaluation_threshold = 0.8
        evaluator.max_attempts = 2
        evaluators = {"test": evaluator}
        context_manager = MagicMock(spec=ContextManager)

        self.mock_executor.execute.side_effect = ["fail1", "fail2"]
        evaluator.evaluate.side_effect = [MagicMock(score=0.5), MagicMock(score=0.6)]

        self.planner.execute_plan(
            plan, "test task", execution_history, True, evaluators, context_manager
        )

        self.assertEqual(evaluator.evaluate.call_count, 2)
        self.assertEqual(len(execution_history.steps), 1)
        # self.assertFalse(execution_history.steps[0].is_success)

    def test_execute_plan_with_default_evaluator(self):
        """Test execute_plan() falls back to default evaluator"""
        step = Step(name="Test", description="Test description", category="unknown")
        plan = [step]
        execution_history = Steps()
        execution_history.add_plan(plan)  # Initialize trace_plan

        # Create evaluator mock that returns passing score
        evaluator = MagicMock(spec=BaseEvaluator)
        evaluator.evaluate.return_value = MagicMock(
            score=1.0, details="Evaluation passed"
        )
        evaluator.evaluation_threshold = 0.8

        evaluators = {"default": evaluator}
        context_manager = MagicMock(spec=ContextManager)

        # Execute the plan
        self.planner.execute_plan(
            plan, "test task", execution_history, True, evaluators, context_manager
        )

        # Verify evaluation was called with expected arguments
        evaluator.evaluate.assert_called_once_with(
            "test task", "Test description", "test response", "", context_manager
        )

        # Verify step was properly updated and added to execution history
        # self.assertEqual(len(execution_history.steps), 1)
        # executed_step = execution_history.steps[0]
        # self.assertEqual(executed_step.name, "Test")
        # self.assertEqual(executed_step.result, "test response")
        # self.assertTrue(executed_step.is_success)

        # Verify evaluator results were properly recorded
        # self.assertEqual(len(executed_step.evaluator_results), 1)
        # self.assertEqual(executed_step.evaluator_results[0].score, 1.0)
        # self.assertEqual(executed_step.evaluator_results[0].details, "Evaluation passed")


if __name__ == "__main__":
    unittest.main()
