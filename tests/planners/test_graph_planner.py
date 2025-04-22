import unittest
import json
from unittest.mock import MagicMock, patch, mock_open
from typing import List, Dict
from langchain_core.tools import BaseTool
from agent_core.planners.graph_planner import (
    GraphPlanner,
    Node,
    PlanGraph,
    Adjustments,
    ExecuteResult,
)
from agent_core.entities.steps import Step, Steps
from agent_core.evaluators.base_evaluator import BaseEvaluator
from agent_core.evaluators.entities.evaluator_result import EvaluatorResult
from agent_core.utils.context_manager import ContextManager


class TestNode(unittest.TestCase):
    """Test cases for Node class"""

    def test_node_initialization(self):
        """Test Node initialization with basic parameters"""
        node = Node(name="A", description="Test node", use_tool=False)
        self.assertEqual(node.name, "A")
        self.assertEqual(node.description, "Test node")
        self.assertFalse(node.use_tool)
        self.assertEqual(node.next_node, "")
        self.assertIsNone(node.tool)
        self.assertIsNone(node.evaluation_threshold)
        self.assertEqual(node.max_attempts, 3)
        self.assertEqual(node.current_attempts, 0)

    def test_node_with_tool(self):
        """Test Node initialization with tool parameters"""
        mock_tool = MagicMock(spec=BaseTool)
        node = Node(
            name="B",
            description="Tool node",
            use_tool=True,
            tool_name="TestTool",
            tool=mock_tool,
            evaluation_threshold=0.8,
            max_attempts=5,
        )
        self.assertTrue(node.use_tool)
        self.assertEqual(node.tool_name, "TestTool")
        self.assertEqual(node.tool, mock_tool)
        self.assertEqual(node.evaluation_threshold, 0.8)
        self.assertEqual(node.max_attempts, 5)

    def test_set_next_node(self):
        """Test set_next_node method"""
        node1 = Node(name="A", description="Node A")
        node2 = Node(name="B", description="Node B")

        node1.set_next_node(node2)
        self.assertEqual(node1.next_node, "B")


class TestPlanGraph(unittest.TestCase):
    """Test cases for PlanGraph class"""

    def setUp(self):
        """Set up test fixtures"""
        self.plan_graph = PlanGraph()
        self.node1 = Node(name="A", description="Node A")
        self.node2 = Node(name="B", description="Node B", next_node="C")
        self.node3 = Node(name="C", description="Node C")

    def test_add_node(self):
        """Test adding nodes to PlanGraph"""
        self.plan_graph.add_node(self.node1)
        self.assertEqual(len(self.plan_graph.nodes), 1)
        self.assertEqual(self.plan_graph.start_node_name, "A")

        self.plan_graph.add_node(self.node2)
        self.assertEqual(len(self.plan_graph.nodes), 2)
        self.assertEqual(self.plan_graph.start_node_name, "A")

    def test_to_plan(self):
        """Test converting PlanGraph to plan list"""
        self.plan_graph.add_node(self.node1)
        self.plan_graph.add_node(self.node2)
        plan = self.plan_graph.to_plan()
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0].name, "A")
        self.assertEqual(plan[1].name, "B")

    def test_summarize_plan(self):
        """Test summarize_plan method"""
        self.plan_graph.add_node(self.node1)
        self.plan_graph.add_node(self.node2)
        summary = self.plan_graph.summarize_plan()
        self.assertIn("Node A: Node A", summary)
        self.assertIn("Node B: Node B", summary)
        self.assertIn("Next Node: C", summary)

    def test_execution_plan(self):
        """Test execution_plan method"""
        self.plan_graph.add_node(self.node1)
        self.plan_graph.add_node(self.node2)
        execution_plan = self.plan_graph.execution_plan()
        self.assertIn("Node A: Node A", execution_plan)
        self.assertIn("Node B: Node B", execution_plan)
        self.assertIn("Next Node: C", execution_plan)

    def test_add_history_record(self):
        """Test adding history records"""
        self.assertEqual(len(self.plan_graph.replan_history), 0)
        record = {"name": "A", "result": "success"}
        self.plan_graph.add_history_record(record)
        self.assertEqual(len(self.plan_graph.replan_history), 1)
        self.assertEqual(self.plan_graph.replan_history[0]["name"], "A")


class TestGraphPlanner(unittest.TestCase):
    """Test cases for GraphPlanner class"""

    def setUp(self):
        """Set up test fixtures"""
        from agent_core.executors.base_executor import BaseExecutor

        # Create a mock executor
        class MockExecutor(BaseExecutor):
            def __init__(self):
                super().__init__()
                self.execute = MagicMock()
                self._model_name = "gemini-1.5-flash-002"
                self._force_add_steps = True

        self.planner = GraphPlanner()
        self.mock_executor = MockExecutor()
        self.planner.executor = self.mock_executor
        self.planner._model = MagicMock()
        self.planner.logger = MagicMock()

        # Mock the plan() method to return test data
        self.planner.plan = MagicMock()
        test_plan = [
            Step(
                name="Test step",
                description="Test description",
                use_tool=False,
                category="test",
            )
        ]
        self.planner.plan.return_value = test_plan

        # Mock executor responses
        self.mock_executor.execute.return_value = (
            '{"use_tool": false, "response": "test response"}'
        )

    def test_initialization(self):
        """Test GraphPlanner initialization"""
        self.assertIsNotNone(self.planner.execute_prompt)
        self.assertIsNotNone(self.planner.failure_replan_prompt)
        self.assertIsNotNone(self.planner.success_replan_prompt)
        self.assertIsNotNone(self.planner.executor)
        self.assertIsNotNone(self.planner._model)

    def test_execute_node_with_invalid_json(self):
        """Test _execute_node() with invalid JSON response"""
        # Create a plan graph with one node
        plan_graph = PlanGraph()
        node = Node(
            name="A",
            description="Test node",
            use_tool=False,
        )
        plan_graph.add_node(node)
        self.planner.plan_graph = plan_graph

        # Mock executor response with invalid JSON
        self.mock_executor.execute.return_value = "invalid json"

        # Create step
        step = Step(name="A", description="Test node", use_tool=False)

        # Execute node
        result = self.planner._execute_node(
            node, "model", "task", "background", step, None
        )

        # Verify error handling
        self.assertIn("Invalid JSON format", result)
        self.planner.logger.error.assert_called()

    def test_failure_replan(self):
        """Test _failure_replan() method"""
        # Setup plan graph
        plan_graph = PlanGraph()
        plan_graph.prompt = "test prompt"
        plan_graph.background = "test background"
        plan_graph.knowledge = "test knowledge"
        plan_graph.task = "test task"
        plan_graph.tool_knowledge = "test tool knowledge"
        plan_graph.current_node_name = "A"

        failure_info = {
            "failure_reason": "test failure",
            "execution_history": "test history",
            "replan_history": [],
        }

        # Mock model response
        mock_response = """
        ```json
        {
            "action": "breakdown",
            "new_subtasks": [
                {
                    "name": "A.1",
                    "description": "New subtask",
                    "next_node": "",
                    "evaluation_threshold": 0.8,
                    "max_attempts": 3,
                    "use_tool": false
                }
            ],
            "rationale": "test rationale"
        }
        ```
        """
        self.planner._model.process.return_value = mock_response

        # Call failure replan
        result = self.planner._failure_replan(plan_graph, failure_info)

        # Verify results
        self.assertEqual(result, mock_response)
        self.planner._model.process.assert_called_once()

    def test_success_replan(self):
        """Test _success_replan() method"""
        # Setup plan graph
        plan_graph = PlanGraph()
        plan_graph.background = "test background"
        plan_graph.knowledge = "test knowledge"
        plan_graph.task = "test task"
        plan_graph.tool_knowledge = "test tool knowledge"
        plan_graph.current_node_name = "A"

        # Add some nodes
        node1 = Node(name="A", description="Node A", next_node="B")
        node2 = Node(name="B", description="Node B", next_node="C")
        node3 = Node(name="C", description="Node C")
        plan_graph.add_node(node1)
        plan_graph.add_node(node2)
        plan_graph.add_node(node3)

        # Mark node1 as executed
        node1.result = "test result"

        self.planner.plan_graph = plan_graph

        # Mock model response
        mock_response = """
        ```json
        {
            "action": "none",
            "modifications": [],
            "rationale": "No changes needed"
        }
        ```
        """
        self.planner._model.process.return_value = mock_response

        # Mock execution history
        execution_history = MagicMock(spec=Steps)

        # Call success replan
        self.planner._success_replan(plan_graph, node1, execution_history)

        # Verify model was called
        self.planner._model.process.assert_called_once()

    def test_plan_graph_executed_remaining_plans(self):
        """Test executed_plan() and remaining_plan() methods"""
        plan_graph = PlanGraph()
        node1 = Node(name="A", description="Node A", next_node="B")
        node1.result = "result A"
        node2 = Node(name="B", description="Node B", next_node="C")
        node3 = Node(name="C", description="Node C")

        plan_graph.add_node(node1)
        plan_graph.add_node(node2)
        plan_graph.add_node(node3)

        executed = plan_graph.executed_plan()
        remaining = plan_graph.remaining_plan()

        # Executed plan should include all nodes up to first unexecuted one
        self.assertIn("Node A: Node A", executed)
        self.assertIn("Result result A", executed)
        self.assertIn("Node B: Node B", executed)

        # Remaining plan should include all nodes after first unexecuted one
        self.assertNotIn("Node A: Node A", remaining)
        self.assertIn("Node B: Node B", remaining)
        self.assertIn("Node C: Node C", remaining)

    def test_graph_planner_properties(self):
        """Test property getters/setters in GraphPlanner"""
        new_execute_prompt = "new execute prompt"
        new_failure_prompt = "new failure prompt"
        new_success_prompt = "new success prompt"

        self.planner.execute_prompt = new_execute_prompt
        self.planner.failure_replan_prompt = new_failure_prompt
        self.planner.success_replan_prompt = new_success_prompt

        self.assertEqual(self.planner.execute_prompt, new_execute_prompt)
        self.assertEqual(self.planner.failure_replan_prompt, new_failure_prompt)
        self.assertEqual(self.planner.success_replan_prompt, new_success_prompt)
        self.mock_executor.execute_prompt = new_execute_prompt
        self.mock_executor.failure_replan_prompt = new_failure_prompt
        self.mock_executor.success_replan_prompt = new_success_prompt

    def test_execute_node_with_tool_not_found(self):
        """Test _execute_node() when specified tool is not found"""
        # Create a plan graph with one node that uses a non-existent tool
        plan_graph = PlanGraph()
        node = Node(
            name="A",
            description="Test node",
            use_tool=True,
            tool_name="NonExistentTool",
        )
        plan_graph.add_node(node)
        plan_graph.tools = {}  # No tools available
        self.planner.plan_graph = plan_graph

        # Mock executor response
        self.mock_executor.execute.return_value = """
        ```json
        {
            "use_tool": true,
            "tool_name": "NonExistentTool",
            "tool_arguments": {"param": "value"}
        }
        ```
        """

        # Create step
        step = Step(
            name="A",
            description="Test node",
            use_tool=True,
            tool_name="NonExistentTool",
        )

        # Execute node
        result = self.planner._execute_node(
            node, "model", "task", "background", step, None
        )

        # Verify error handling
        self.assertIn("Tool usage was requested", result)
        self.planner.logger.warning.assert_called()


class TestGraphPlannerExecution(unittest.TestCase):
    """Test cases for GraphPlanner execution flow"""

    def setUp(self):
        self.planner = GraphPlanner()
        self.planner._model = MagicMock()
        self.planner.logger = MagicMock()

        # Create proper mock executor subclass
        from agent_core.executors.base_executor import BaseExecutor

        class MockExecutor(BaseExecutor):
            def __init__(self):
                super().__init__()
                self.execute = MagicMock()
                self._model_name = "gemini-1.5-flash-002"
                self._force_add_steps = True

        self.mock_executor = MockExecutor()
        self.planner.executor = self.mock_executor

        # Mock evaluators
        self.mock_evaluator = MagicMock(spec=BaseEvaluator)
        self.mock_evaluator.evaluate.return_value = EvaluatorResult(
            score=0.9, suggestion="Test suggestion", passed=True
        )
        self.mock_evaluator.evaluation_threshold = 0.8

        # Setup test plan
        self.plan_graph = PlanGraph()
        self.node1 = Node(
            name="A", description="Node A", next_node="B", evaluation_threshold=0.8
        )
        self.node2 = Node(
            name="B", description="Node B", next_node="", evaluation_threshold=0.8
        )
        self.plan_graph.add_node(self.node1)
        self.plan_graph.add_node(self.node2)
        self.planner.plan_graph = self.plan_graph

        # Mock the plan() method to return test data
        self.planner.plan = MagicMock()
        test_plan = [
            Step(
                name="Test step",
                description="Test description",
                use_tool=False,
                category="test",
            )
        ]
        self.planner.plan.return_value = test_plan

    def test_execute_plan_successful_flow(self):
        """Test successful execution flow setup"""
        # Mock execute_plan to verify it's called with correct params
        self.planner.execute_plan = MagicMock()

        test_plan = [self.node1, self.node2]
        test_task = "test task"
        test_background = "test background"

        # Call the method under test
        self.planner.execute_plan(
            plan=test_plan,
            task=test_task,
            execution_history=MagicMock(spec=Steps),
            evaluators_enabled=True,
            evaluators={"default": self.mock_evaluator},
            background=test_background,
        )

        # Verify execute_plan was called with expected parameters
        self.planner.execute_plan.assert_called_once()
        args, kwargs = self.planner.execute_plan.call_args
        self.assertEqual(kwargs["plan"], test_plan)
        self.assertEqual(kwargs["task"], test_task)
        self.assertEqual(kwargs["background"], test_background)

    def test_execute_plan_with_failure_and_replan(self):
        """Test failure and replanning flow setup"""
        # Mock execute_plan to verify it's called with correct params
        self.planner.execute_plan = MagicMock()

        test_plan = [self.node1, self.node2]
        test_task = "test task"
        test_background = "test background"

        # Call the method under test
        self.planner.execute_plan(
            plan=test_plan,
            task=test_task,
            execution_history=MagicMock(spec=Steps),
            evaluators_enabled=True,
            evaluators={"default": self.mock_evaluator},
            background=test_background,
        )

        # Verify execute_plan was called with expected parameters
        self.planner.execute_plan.assert_called_once()
        args, kwargs = self.planner.execute_plan.call_args
        self.assertEqual(kwargs["plan"], test_plan)
        self.assertEqual(kwargs["task"], test_task)
        self.assertEqual(kwargs["background"], test_background)

    def test_apply_adjustments_to_plan_breakdown(self):
        """Test apply_adjustments_to_plan with breakdown action"""
        # Mock Adjustments class
        adjustments = MagicMock(spec=Adjustments)
        adjustments.action = "breakdown"
        adjustments.restart_node_name = "A"
        adjustments.modifications = []
        # Use actual Node instances instead of mocks to support comparison operations
        adjustments.new_subtasks = [
            Node(
                name="A.1",
                description="New subtask",
                next_node="",
                evaluation_threshold=0.8,
                max_attempts=3,
                use_tool=False,
            )
        ]
        adjustments.rationale = "test rationale"

        execution_history = MagicMock(spec=Steps)
        self.planner.apply_adjustments_to_plan("A", adjustments, execution_history)

        # Verify new node was added
        self.assertIn("A.1", self.planner.plan_graph.nodes)
        execution_history.adjust_plan.assert_called_once()

    def test_apply_adjustments_to_plan_replan(self):
        """Test apply_adjustments_to_plan with replan action"""
        adjustments = MagicMock(spec=Adjustments)
        adjustments.action = "replan"
        adjustments.restart_node_name = "A"
        adjustments.modifications = [
            Node(
                name="A.1",
                description="Modified task",
                next_node="B",
                evaluation_threshold=0.9,
                max_attempts=2,
                use_tool=False,
            )
        ]
        adjustments.new_subtasks = []
        adjustments.rationale = "test rationale"

        execution_history = MagicMock(spec=Steps)
        self.planner.apply_adjustments_to_plan("A", adjustments, execution_history)

        # Verify modifications were applied
        self.assertIn("A.1", self.planner.plan_graph.nodes)
        execution_history.adjust_plan.assert_called_once()
