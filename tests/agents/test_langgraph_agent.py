import pytest
from unittest.mock import MagicMock, patch
from agent_core.agents.langgraph_agent import (
    AgentState,
    process_state_schema,
    get_task,
    set_response,
    agent_execute,
)
from agent_core.agents import Agent
from agent_core.planners.base_planner import BasePlanner
from langchain_core.tools import BaseTool
from agent_core.evaluators import BaseEvaluator


class TestAgentState:
    def test_agent_state_typing(self):
        """Test that AgentState has all expected fields with correct types"""
        state = AgentState(
            messages=[{"content": "test"}],
            planner=MagicMock(spec=BasePlanner),
            tools=[MagicMock(spec=BaseTool)],
            knowledge="test knowledge",
            background="test background",
            model_name="test_model",
            log_level="INFO",
            evaluators_enabled=True,
            evaluators={"test": MagicMock(spec=BaseEvaluator)},
            task_function=lambda x: x,
            response_function=lambda x, y: y,
        )
        assert state["messages"][0]["content"] == "test"
        assert isinstance(state["planner"], BasePlanner)
        assert isinstance(state["tools"][0], BaseTool)


class TestProcessStateSchema:
    def test_process_state_schema_all_fields(self):
        """Test processing state schema with all possible fields"""
        mock_agent = MagicMock(spec=Agent)
        state_schema = {
            "planner": MagicMock(spec=BasePlanner),
            "tools": [MagicMock(spec=BaseTool)],
            "knowledge": "test knowledge",
            "background": "test background",
            "evaluators_enabled": True,
            "evaluators": {"test": MagicMock(spec=BaseEvaluator)},
        }

        process_state_schema(mock_agent, state_schema)

        assert mock_agent.planner == state_schema["planner"]
        assert mock_agent.tools == state_schema["tools"]
        assert mock_agent.knowledge == state_schema["knowledge"]
        assert mock_agent.background == state_schema["background"]
        assert mock_agent.evaluators_enabled == state_schema["evaluators_enabled"]
        assert mock_agent.evaluators == state_schema["evaluators"]

    def test_process_state_schema_partial_fields(self):
        """Test processing state schema with only some fields"""
        # Create a real Agent instance instead of MagicMock
        agent = Agent(model_name=None, log_level=None)
        state_schema = {
            "knowledge": "partial knowledge",
            "background": "partial background",
        }

        process_state_schema(agent, state_schema)

        assert agent.knowledge == state_schema["knowledge"]
        assert agent.background == state_schema["background"]
        assert agent.planner is None  # Not set in state_schema
        assert agent.tools is None  # Not set in state_schema


class TestGetTask:
    def test_get_task_with_task_function(self):
        """Test get_task when task_function is provided"""
        state_schema = {
            "task_function": lambda x: "custom task",
            "messages": [{"content": "message task"}],
        }
        assert get_task(state_schema) == "custom task"

    def test_get_task_with_messages(self):
        """Test get_task when messages are provided"""
        state_schema = {"messages": [{"content": "message task"}]}
        assert get_task(state_schema) == "message task"

    def test_get_task_empty(self):
        """Test get_task with empty state"""
        assert get_task({}) is None


class TestSetResponse:
    def test_set_response_with_response_function(self):
        """Test set_response when response_function is provided"""
        state_schema = {
            "response_function": lambda x, y: {"custom": y},
            "messages": [{"content": "test"}],
        }
        result = set_response(state_schema, "test response")
        assert result == {"custom": "test response"}

    def test_set_response_with_messages(self):
        """Test set_response when messages are provided"""
        state_schema = {"messages": [{"content": "test"}]}
        result = set_response(state_schema, "test response")
        assert result == {"messages": "test response"}

    def test_set_response_empty(self):
        """Test set_response with empty state"""
        assert set_response({}, "test") is None


class TestAgentExecute:
    @patch("agent_core.agents.langgraph_agent.Agent")
    def test_agent_execute_full_state(self, mock_agent_class):
        """Test agent_execute with full state schema"""
        mock_agent = MagicMock()
        mock_agent.execute.return_value = "test response"
        mock_agent_class.return_value = mock_agent

        state_schema = {
            "model_name": "test_model",
            "log_level": "DEBUG",
            "planner": MagicMock(),
            "task_function": lambda x: "custom task",
            "response_function": lambda x, y: {"custom": y},
        }

        result = agent_execute(state_schema)
        assert result == {"custom": "test response"}
        mock_agent_class.assert_called_with("test_model", "DEBUG")
        mock_agent.execute.assert_called_with("custom task")

    @patch("agent_core.agents.langgraph_agent.Agent")
    def test_agent_execute_minimal_state(self, mock_agent_class):
        """Test agent_execute with minimal state schema"""
        mock_agent = MagicMock()
        mock_agent.execute.return_value = "test response"
        mock_agent_class.return_value = mock_agent

        state_schema = {"messages": [{"content": "message task"}]}

        result = agent_execute(state_schema)
        assert result == {"messages": "test response"}
        mock_agent_class.assert_called_with(None, None)
        mock_agent.execute.assert_called_with("message task")

    def test_agent_execute_empty_state(self):
        """Test agent_execute with empty state"""
        with pytest.raises(Exception):
            agent_execute(None)
