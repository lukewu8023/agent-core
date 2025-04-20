import unittest
from unittest.mock import MagicMock, patch
from typing import List, Optional
from langchain_core.tools import BaseTool
from agent_core.planners.base_planner import BasePlanner, tool_knowledge_format
from agent_core.executors.base_executor import BaseExecutor


class TestBasePlanner(unittest.TestCase):
    """Test cases for BasePlanner class"""

    def setUp(self):
        """Set up test fixtures"""

        # Create a mock subclass of BasePlanner to test abstract methods
        class MockPlanner(BasePlanner):
            def plan(self, task, tools, knowledge="", background="", categories=None):
                return []

            def execute_plan(self, *args, **kwargs):
                return None

        self.planner = MockPlanner()
        self.mock_executor = MagicMock(spec=BaseExecutor)

    def test_tool_knowledge_format(self):
        """Test tool_knowledge_format function"""
        # Test with no tools
        self.assertEqual(tool_knowledge_format(None), "")

        # Test with empty tools list
        self.assertEqual(tool_knowledge_format([]), "")

        # Test with mock tools
        mock_tool = MagicMock(spec=BaseTool)
        mock_args_schema = MagicMock()
        mock_args_schema.model_json_schema.return_value = {"test": "schema"}
        mock_tool.args_schema = mock_args_schema
        tools = [mock_tool]
        result = tool_knowledge_format(tools)
        self.assertEqual(result, "{'test': 'schema'}")

    def test_initialization(self):
        """Test BasePlanner initialization"""
        from agent_core.planners.base_planner import DEFAULT_PROMPT

        self.assertEqual(self.planner.prompt, DEFAULT_PROMPT)
        self.assertIsInstance(self.planner.executor, BaseExecutor)

    def test_model_name_property(self):
        """Test model_name property and setter"""
        # Test initial model_name
        self.assertEqual(self.planner.model_name, "gemini-1.5-flash-002")

        # Test setting model_name
        test_name = "gemini-1.5-flash-002"  # Use valid model name
        self.planner.model_name = test_name
        self.assertEqual(self.planner.model_name, test_name)
        self.assertEqual(self.planner.executor.model_name, test_name)

    def test_executor_property(self):
        """Test executor property and setter"""
        # Test initial executor
        self.assertIsInstance(self.planner.executor, BaseExecutor)

        # Test setting executor
        self.planner.executor = self.mock_executor
        self.assertEqual(self.planner.executor, self.mock_executor)

        # Test invalid executor type
        with self.assertRaises(TypeError):
            self.planner.executor = "invalid_executor"

    def test_configure_executor(self):
        """Test _configure_executor method"""
        # Should not raise any exceptions
        self.planner._configure_executor()

    @patch.object(BasePlanner, "_configure_executor")
    def test_executor_setter_calls_configure(self, mock_configure):
        """Test executor setter calls _configure_executor"""
        self.planner.executor = self.mock_executor
        mock_configure.assert_called_once()


if __name__ == "__main__":
    unittest.main()
