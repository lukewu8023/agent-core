import pytest
from unittest.mock import MagicMock, patch, mock_open

from agent_core.agents.agent import Agent
from agent_core.planners.base_planner import BasePlanner
from agent_core.evaluators import BaseEvaluator
from agent_core.entities.steps import Steps, Step, Summary
from agent_core.utils.context_manager import ContextManager


# Test Fixtures
@pytest.fixture
def mock_model():
    model = MagicMock()
    model.process.return_value = "mock response"
    return model


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setattr(
        "agent_core.models.base_model.Environment",
        lambda: type("MockEnv", (), {"agent_core_log_level": "INFO"}),
    )


@pytest.fixture
def agent(mock_model):
    with patch("agent_core.models.model_registry.ModelRegistry.load_models"), patch(
            "agent_core.models.model_registry.ModelRegistry.get_model",
            return_value=mock_model,
    ):
        agent = Agent()
        agent.tools = [MagicMock()]
        return agent


@pytest.fixture
def mock_planner():
    planner = MagicMock(spec=BasePlanner)
    planner.plan.return_value = MagicMock()
    planner.execute_plan.return_value = "mock plan result"
    return planner


@pytest.fixture
def mock_evaluator():
    evaluator = MagicMock(spec=BaseEvaluator)
    return evaluator


# Test Cases
class TestAgentInitialization:
    def test_agent_init(self, agent):
        assert agent is not None
        assert agent.planner is None
        assert isinstance(agent.context, ContextManager)
        assert agent.execute_prompt is not None
        assert agent.summary_prompt is not None
        assert agent.response_prompt is not None


class TestAgentExecution:
    def test_execute_without_planner(self, agent, mock_model):
        task = "test task"
        # Initialize trace_plan before execution
        agent._execution_history.trace_plan = {1: MagicMock(plan=[MagicMock()])}
        result = agent.execute(task)
        assert result == "mock response"
        mock_model.process.assert_called_once()
        assert len(agent.execution_history.steps) > 0

    def test_execute_with_planner(self, agent, mock_model, mock_planner):
        mock_model.process.return_value = '{"summary": "test"}'
        agent.planner = mock_planner
        task = "test task"
        result = agent.execute(task)
        assert result is not None
        mock_planner.plan.assert_called_once()
        mock_planner.execute_plan.assert_called_once()

    def test_execute_without_planner_uses_correct_prompt(self, agent, mock_model):
        task = "test task"
        # Initialize trace_plan before execution
        agent._execution_history.trace_plan = {1: MagicMock(plan=[MagicMock()])}
        agent.execute(task)
        prompt = mock_model.process.call_args[0][0]
        assert task in prompt
        assert agent.background in prompt


class TestPlannerManagement:
    def test_set_planner(self, agent, mock_planner):
        agent.planner = mock_planner
        assert agent.planner == mock_planner


class TestEvaluatorManagement:
    def test_enable_disable_evaluators(self, agent):
        agent.enable_evaluators()
        assert agent.evaluators_enabled is True
        agent.disable_evaluators()
        assert agent.evaluators_enabled is False

    def test_add_evaluator(self, agent, mock_evaluator):
        category = "test_category"
        agent.add_evaluator(category, mock_evaluator)
        assert category in agent.evaluators
        assert agent.evaluators[category] == mock_evaluator

    def test_update_evaluator(self, agent, mock_evaluator):
        category = "test_category"
        agent.add_evaluator(category, mock_evaluator)
        new_evaluator = MagicMock(spec=BaseEvaluator)
        agent.update_evaluator(category, new_evaluator)
        assert agent.evaluators[category] == new_evaluator


class TestExecutionHistory:
    def test_execution_history_property(self, agent):
        assert isinstance(agent.execution_history, Steps)

    def test_export_execution_trace(self, agent):
        agent._execution_history = MagicMock()
        agent._execution_history.model_dump.return_value = {"test": "data"}

        with patch("builtins.open", mock_open()) as mock_file, patch(
                "os.makedirs"
        ), patch("json.dump"):
            agent.export_execution_trace()
            mock_file.assert_called()


class TestResultGeneration:
    def test_get_execution_result_summary(self, agent, mock_model):
        mock_model.process.return_value = '{"summary": "test"}'
        agent._execution_history = MagicMock()
        agent._execution_history.execution_history_to_str.return_value = "test history"
        summary = Summary(summary='test', output_result='', conclusion='')
        result = agent.get_execution_result_summary()
        assert summary == result
        mock_model.process.assert_called_once()

    def test_get_final_response(self, agent, mock_model):
        task = "test task"
        agent._execution_history = MagicMock()
        agent._execution_history.execution_history_to_str.return_value = "test history"

        result = agent.get_final_response(task)
        assert result == "mock response"
        mock_model.process.assert_called_once()

    def test_get_execution_reasoning_no_history(self, agent):
        result = agent.get_execution_reasoning()
        assert [] == result

    def test_get_execution_reasoning_with_history(self, agent):
        agent._execution_history = MagicMock()
        agent._execution_history.trace_plan = {
            1: MagicMock(plan=[MagicMock(name="step1", description="desc1")])
        }
        agent._execution_history.trace_steps = [
            MagicMock(action="success", name="step1", use_tool=False, description="desc1")
        ]

        result = agent.get_execution_reasoning()
        assert result is not None


class TestTokenManagement:
    def test_get_token(self, agent):
        with patch(
                "agent_core.models.model_registry.ModelRegistry.get_token",
                return_value=(10, 20),
        ):
            agent.get_token()
            assert agent._execution_history.input_tokens == 10
            assert agent._execution_history.output_tokens == 20


class TestEvaluatorManagement:
    def test_load_default_evaluators(self, agent):
        agent._load_default_evaluators()
        assert isinstance(agent.evaluators, dict)
        assert len(agent.evaluators) > 0


class TestExecutionHistoryExtended:
    def test_get_execution_history_method(self, agent):
        agent._execution_history = MagicMock()
        agent._execution_history.execution_history_to_str.return_value = "test"
        agent.get_execution_history()
        agent._execution_history.execution_history_to_str.assert_called_once()

    def test_execution_responses_property(self, agent):
        agent._execution_history = MagicMock()
        agent._execution_history.execution_history_to_responses.return_value = "test"
        result = agent.execution_responses
        assert result == "test"
        agent._execution_history.execution_history_to_responses.assert_called_once()

    def test_export_execution_trace_no_history(self, agent):
        with patch("builtins.open", mock_open()) as mock_file, patch(
                "os.makedirs"
        ), patch("json.dump"):
            agent._execution_history = Steps()  # Empty history
            agent.export_execution_trace()
            mock_file.assert_called()


class TestPlanAdjustment:
    def test_add_retry_step(self, agent, mock_planner):
        agent.planner = mock_planner
        task = "test task"
        # Mock a failed step
        agent._execution_history.add_plan([Step(name="test", description="test")])
        agent._execution_history.add_failure_step(Step(name="test", description="test"))
        # Add retry
        agent._execution_history.add_retry_step(Step(name="retry", description="retry"))
        assert len(agent._execution_history.trace_steps) == 2

    def test_adjust_plan(self, agent, mock_planner):
        agent.planner = mock_planner
        task = "test task"
        # Initial setup
        agent._execution_history.add_plan([Step(name="initial", description="initial")])
        # Adjust plan
        agent._execution_history.adjust_plan(
            "adjust", [Step(name="adjusted", description="adjusted")], "adjustment"
        )
        assert len(agent._execution_history.trace_plan) == 2


class TestEvaluatorEdgeCases:
    def test_update_nonexistent_evaluator(self, agent, mock_evaluator):
        agent.update_evaluator("nonexistent", mock_evaluator)
        assert "nonexistent" in agent.evaluators


class TestExecutionHistoryEdgeCases:
    def test_get_execution_reasoning_empty_plan(self, agent):
        agent._execution_history.trace_plan = {1: MagicMock(plan=[])}
        result = agent.get_execution_reasoning()
        assert "No execution reasoning available" not in result

    def test_get_execution_reasoning_with_retries(self, agent):
        agent._execution_history.trace_plan = {
            1: MagicMock(plan=[MagicMock(name="step1", description="desc1")])
        }
        retry_step = MagicMock(name="retry", description="retry desc", action="retry")
        agent._execution_history.trace_steps = [
            MagicMock(name="step1", description="desc1", action="failure"),
            retry_step,
        ]
        result = agent.get_execution_reasoning()
        assert "retry desc" in result


class TestNarrativeGeneration:
    def test_get_plan_narrative_default_template(self, agent):
        result = agent._get_plan_narrative(plan_steps="test steps", step=None)
        assert "test steps" in result

    def test_get_plan_narrative_with_step(self, agent):
        mock_step = MagicMock()
        mock_step.action = "failure replan"
        result = agent._get_plan_narrative(plan_steps="test steps", step=mock_step)
        assert "test steps" in result


class TestExecutionHistoryEdgeCases:
    def test_get_last_step_output_with_steps(self, agent):
        mock_step = MagicMock()
        mock_step.result = "test_result"
        agent._execution_history.steps = [mock_step]
        result = agent._execution_history.get_last_step_output()
        assert result.result == "test_result"

    def test_get_last_step_output_empty(self, agent):
        agent._execution_history.steps = []
        result = agent._execution_history.get_last_step_output()
        assert result == ""
