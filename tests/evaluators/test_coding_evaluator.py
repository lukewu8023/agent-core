import json
import pytest
from unittest.mock import MagicMock, patch
from agent_core.evaluators.coding_evaluator import (
    CodingEvaluator,
    generate_improvement_suggestions,
    reject_code
)
from agent_core.evaluators.entities.evaluator_result import EvaluatorResult

class TestGenerateImprovementSuggestions:
    """Tests for the generate_improvement_suggestions helper function"""
    
    def test_no_low_scores(self):
        """Test when no scores are below 3"""
        scores = [
            ("Requirements Coverage", 4),
            ("Correctness", 3),
            ("Code Style and Conventions", 5)
        ]
        result = generate_improvement_suggestions(scores)
        assert "Review the overall implementation" in result

    def test_with_low_scores(self):
        """Test when some scores are below 3"""
        scores = [
            ("Requirements Coverage", 2),
            ("Correctness", 1),
            ("Code Style and Conventions", 4)
        ]
        result = generate_improvement_suggestions(scores)
        assert "Requirements Coverage" in result
        assert "Correctness" in result
        assert "Code Style and Conventions" not in result

class TestCodingEvaluator:
    """Tests for the CodingEvaluator class"""
    
    def test_init_defaults(self):
        """Test initialization with default parameters"""
        evaluator = CodingEvaluator()
        assert evaluator.evaluation_threshold == 0.8
        assert evaluator.model_name == "gemini-1.5-flash-002"

    def test_init_custom(self):
        """Test initialization with custom parameters"""
        evaluator = CodingEvaluator(
            model_name="gpt-3.5-turbo",
            evaluation_threshold=0.9
        )
        assert evaluator.evaluation_threshold == 0.9
        assert evaluator.model_name == "gpt-3.5-turbo"

    def test_default_prompt(self):
        """Test the default prompt content"""
        evaluator = CodingEvaluator()
        prompt = evaluator.default_prompt()
        assert "You are an expert code reviewer" in prompt
        assert "Requirements Coverage" in prompt
        assert "Evaluation (JSON Format):" in prompt

    @patch('agent_core.evaluators.coding_evaluator.CodingEvaluator.parse_scored_evaluation_response')
    def test_evaluate_success(self, mock_parse):
        """Test successful evaluation with mock model response"""
        evaluator = CodingEvaluator()
        evaluator._model = MagicMock()
        evaluator._model.process.return_value = '{"decision": "Accept Code", "scores": []}'
        
        # Mock returns an EvaluatorResult with proper initialization
        mock_parse.return_value = EvaluatorResult(
            decision="Accept Code",
            score=20,
            details={
                "score_breakdown": [],
                "raw_evaluation": "",
                "total_applicable_score": 20,
                "improvement_suggestions": ""
            }
        )
        
        result = evaluator.evaluate(
            root_task="test task",
            request="test request",
            response="test response",
            background="test background",
            context_manager=MagicMock()
        )
        
        assert result.decision == "Accept Code"
        assert result.score == 20
        assert isinstance(result.details, dict)

    def test_parse_scored_evaluation_response_invalid_json(self):
        """Test parsing invalid JSON response"""
        evaluator = CodingEvaluator()
        result = evaluator.parse_scored_evaluation_response("invalid json")
        assert result[0] == reject_code
        assert result[1] == 0
        assert result[2] == []

    def test_evaluate_model_error(self):
        """Test evaluation when model processing fails"""
        evaluator = CodingEvaluator()
        evaluator._model = MagicMock()
        evaluator._model.process.side_effect = Exception("Model error")
        
        result = evaluator.evaluate(
            root_task="test task",
            request="test request",
            response="test response",
            background="test background",
            context_manager=MagicMock()
        )
        
        assert result.decision == reject_code
        assert result.score == 0
        assert "Model evaluation failed" in result.details["improvement_suggestions"]

    def test_generate_improvement_suggestions_empty(self):
        """Test improvement suggestions with empty scores"""
        result = generate_improvement_suggestions([])
        assert "Review the overall implementation" in result
