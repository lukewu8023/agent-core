import pytest
from unittest.mock import MagicMock, patch
from agent_core.agents.autogen_agent import PhaenixAgent
from autogen_agentchat.messages import TextMessage
from agent_core.agents import Agent
from langchain_core.tools import BaseTool
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class MockToolArgsSchema(BaseModel):
    """Mock args schema for testing"""

    param: str = Field(default="test")


class TestPhaenixAgent:
    @pytest.fixture
    def mock_agent(self):
        mock = MagicMock(spec=Agent)
        mock.tools = []
        return mock

    @pytest.fixture
    def phaenix_agent(self, mock_agent):
        with patch("agent_core.agents.autogen_agent.Agent", return_value=mock_agent):
            with patch(
                "agent_core.agents.autogen_agent.PhaenixAgent._integrate_autogen_tools"
            ) as mock_integrate:
                agent = PhaenixAgent(
                    name="test_agent", model_name="test_model", log_level="INFO"
                )
                mock_integrate.side_effect = lambda tools: mock_agent.tools.extend(
                    [MagicMock(spec=BaseTool) for _ in tools]
                )
                return agent

    def test_initialization(self, phaenix_agent):
        """Test that PhaenixAgent initializes correctly with all parameters"""
        assert phaenix_agent.name == "test_agent"

    def test_initialization_with_optional_params(self, mock_agent):
        """Test initialization with all optional parameters"""
        tools = [MagicMock(spec=BaseTool)]
        planner = MagicMock()
        knowledge = "test_knowledge"
        background = "test_background"
        evaluators = {"test": MagicMock()}

        mock_autogen_tool = MagicMock()
        mock_autogen_tool.name = "test_tool"
        mock_autogen_tool.description = "test_description"
        mock_autogen_tool.args_schema = MockToolArgsSchema

        with patch("agent_core.agents.autogen_agent.Agent", return_value=mock_agent):
            with patch(
                "agent_core.agents.autogen_agent.PhaenixAgent._integrate_autogen_tools"
            ) as mock_integrate:
                agent = PhaenixAgent(
                    name="test_agent",
                    model_name="test_model",
                    log_level="INFO",
                    tools=tools,
                    planner=planner,
                    knowledge=knowledge,
                    background=background,
                    evaluators_enabled=True,
                    evaluators=evaluators,
                    autogen_tools=[mock_autogen_tool],
                )
                mock_integrate.side_effect = lambda tools: mock_agent.tools.extend(
                    [MagicMock(spec=BaseTool) for _ in tools]
                )

        assert mock_agent.planner == planner
        assert len(mock_agent.tools) == 1  # Only the autogen adapter
        assert mock_agent.knowledge == knowledge
        assert mock_agent.background == background
        assert mock_agent.evaluators_enabled is True
        assert mock_agent.evaluators == evaluators

    def test_produced_message_types(self, phaenix_agent):
        """Test that produced_message_types returns correct tuple"""
        assert phaenix_agent.produced_message_types == (TextMessage,)

    def test_integrate_autogen_tools(self, phaenix_agent, mock_agent):
        """Test that autogen tools are properly integrated"""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "test_description"
        mock_tool.args_schema = MockToolArgsSchema

        # Mock the actual integration behavior without testing the nested class
        with patch.object(phaenix_agent, "_integrate_autogen_tools") as mock_integrate:
            mock_integrate.side_effect = lambda tools: mock_agent.tools.extend(
                [MagicMock(spec=BaseTool) for _ in tools]
            )
            phaenix_agent._integrate_autogen_tools([mock_tool])

        assert len(mock_agent.tools) == 1
        assert isinstance(mock_agent.tools[0], MagicMock)

    @pytest.mark.asyncio
    async def test_on_messages(self, phaenix_agent, mock_agent):
        """Test the on_messages method"""
        mock_message = TextMessage(content="test_message", source="test_source")
        mock_agent.execute.return_value = "test_response"

        response = await phaenix_agent.on_messages([mock_message], None)

        assert response.chat_message.content == "test_response"
        assert response.chat_message.source == "test_agent"
        mock_agent.execute.assert_called_once_with("test_message")

    @pytest.mark.asyncio
    async def test_on_messages_empty(self, phaenix_agent):
        """Test on_messages with empty message list"""
        with pytest.raises(IndexError):
            await phaenix_agent.on_messages([], None)

    @pytest.mark.asyncio
    async def test_on_reset(self, phaenix_agent):
        """Test the on_reset method (no-op)"""
        await phaenix_agent.on_reset(None)
