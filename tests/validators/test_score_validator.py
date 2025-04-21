# tests/evaluator/test_score_evaluator.py

import pytest
from agent_core.evaluators import GenericEvaluator
from agent_core.evaluators.base_evaluator import parse_scored_evaluation_response
from agent_core.models.model_registry import ModelRegistry
from agent_core.utils.context_manager import ContextManager


@pytest.fixture
def mock_model_name():
    return "gpt-4o-mini"


@pytest.fixture
def mock_model(mock_model_name):
    return ModelRegistry.get_model(mock_model_name)  # or a mock/fake model


@pytest.fixture
def score_evaluator(mock_model_name):
    return GenericEvaluator(mock_model_name)

def test_parse_scored_evaluation_response_1(score_evaluator):
    evaluation_response = """
    {
        "points": [
            {
                "criterion": "Accuracy",
                "score": 4,
                "justification": "The output accurately identifies the components of a flower (petals, stem, leaves) and assigns characters to each part. However, it could be more specific about the arrangement of the characters in relation to each other"
            },
            {
                "criterion": "Completeness",
                "score": 3,
                "justification": "While the output includes the necessary components of a flower, it lacks details on how the characters are arranged spatially to form the shape of a flower. More information on the positioning would enhance completeness."
            },
            {
                "criterion": "Relevance",
                "score": 5,
                "justification": "The content is directly relevant to the subtask of arranging characters to form a flower shape. There are no extraneous details."
            },
            {
                "criterion": "Coherence and Clarity",
                "score": 4,
                "justification": "The output is generally clear and logically structured, but the lack of spatial arrangement details makes it slightly less coherent in terms of visualizing the flower shape."
            },
            {
                "criterion": "Consistency",
                "score": 5,
                "justification": "The output is consistent with the requirements of the subtask and does not contradict any previous information."
            },
            {
                "criterion": "Following Instructions",
                "score": 4,
                "justification": "The output follows the instructions well but could improve by providing a more detailed arrangement that visually represents a flower."
            },
            {
                "criterion": "Error Analysis",
                "score": 5,
                "justification": "The output is free from grammatical, factual, and logical errors."
            },
            {
                "criterion": "Ethical Compliance",
                "score": 5,
                "justification": "The content complies with ethical guidelines and does not contain any inappropriate material."
            }
        ],
    "improvement_suggestion": "Provide a more detailed description of how the characters are arranged spatially to visually represent a flower shape. For example, specify the positioning of the petals around the stem and how the leaves are placed in relation to the stem. This would enhance both completeness and clarity."
    }
   """
    recommendation, numeric_score, improvement_suggestion, details = parse_scored_evaluation_response(
        0.9, evaluation_response
    )
    assert recommendation in ("Accept Output", "Rerun Subtask")
    assert numeric_score == 35/40
