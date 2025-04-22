import re
import json
from unittest.mock import MagicMock, patch
import pytest
from agent_core.utils.llm_chat import LLMChat, _parse_section, _parse_rating


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def llm_chat(mock_llm):
    with patch("agent_core.utils.llm_chat.AgentBasic.__init__", return_value=None):
        chat = LLMChat()
        chat._model = mock_llm
        chat.logger = MagicMock()
        return chat


class TestHelperFunctions:
    def test_parse_section_found(self):
        """Test _parse_section when section is found"""
        text = "Summary: This is a test summary\nRating: 5"
        result = _parse_section(text, "summary")
        assert result == "This is a test summary"

    def test_parse_section_not_found(self):
        """Test _parse_section when section is missing"""
        text = "No summary here"
        result = _parse_section(text, "summary")
        assert result == "No summary found."

    def test_parse_rating_found(self):
        """Test _parse_rating when rating is found"""
        text = "Rating: 7"
        result = _parse_rating(text)
        assert result == 7

    def test_parse_rating_not_found(self):
        """Test _parse_rating when rating is missing"""
        text = "No rating here"
        result = _parse_rating(text)
        assert result == 1

    def test_parse_rating_clamping(self):
        """Test _parse_rating clamps values to 1-10 range"""
        assert _parse_rating("Rating: 0") == 1
        assert _parse_rating("Rating: 11") == 10


class TestLLMChat:
    def test_process(self, llm_chat, mock_llm):
        """Test process method forwards to model and returns response"""
        mock_llm.process.return_value = "test response"
        result = llm_chat.process("test input")
        assert result == "test response"
        mock_llm.process.assert_called_once_with("test input")
        llm_chat.logger.debug.assert_called_once_with("Response: test response")

    def test_evaluate_text(self, llm_chat, mock_llm):
        """Test evaluate_text returns proper dict structure"""
        mock_response = """Summary: Test summary
Rating: 7
Suggestions: Test suggestions"""
        mock_llm.process.return_value = mock_response

        result = llm_chat.evaluate_text("input text", "criteria", 8)

        assert result == {
            "decision": "Fail",
            "rating": 7,
            "summary": "Test summary",
            "suggestions": "Test suggestions",
            "raw_response": mock_response,
        }
        llm_chat.logger.debug.assert_called_once_with(
            f"Evaluate raw response: {mock_response}"
        )

    def test_evaluate_text_no_suggestions(self, llm_chat, mock_llm):
        """Test evaluate_text when suggestions are missing"""
        mock_response = "Summary: Test summary\nRating: 9"
        mock_llm.process.return_value = mock_response

        result = llm_chat.evaluate_text("input text", "criteria", 8)

        assert result["suggestions"] == "No suggestions found."

    def test_parse_llm_response_valid_json(self, llm_chat):
        """Test parse_llm_response with valid JSON"""
        test_json = '{"key": "value"}'
        result = llm_chat.parse_llm_response(f"```json\n{test_json}\n```")
        assert result == {"key": "value"}

    def test_parse_llm_response_invalid_json(self, llm_chat):
        """Test parse_llm_response with invalid JSON"""
        result = llm_chat.parse_llm_response("invalid json")
        assert result is None
        llm_chat.logger.error.assert_called_once()

    def test_evaluate_text_prompt_property(self, llm_chat):
        """Test evaluate_text_prompt getter/setter"""
        assert llm_chat.evaluate_text_prompt == llm_chat._evaluate_text_prompt

        new_prompt = "new prompt"
        llm_chat.evaluate_text_prompt = new_prompt
        assert llm_chat._evaluate_text_prompt == new_prompt
        assert llm_chat.evaluate_text_prompt == new_prompt

    def test_evaluate_text_pass_decision(self, llm_chat, mock_llm):
        """Test evaluate_text returns Pass when rating meets threshold"""
        mock_response = "Summary: Good\nRating: 8"
        mock_llm.process.return_value = mock_response

        result = llm_chat.evaluate_text("input", "criteria", 8)
        assert result["decision"] == "Pass"

    def test_evaluate_text_edge_cases(self, llm_chat, mock_llm):
        """Test evaluate_text with edge case ratings"""
        mock_llm.process.return_value = "Rating: 1"
        assert llm_chat.evaluate_text("input", "criteria", 1)["decision"] == "Pass"

        mock_llm.process.return_value = "Rating: 10"
        assert llm_chat.evaluate_text("input", "criteria", 10)["decision"] == "Pass"

    def test_parse_llm_response_with_backticks(self, llm_chat):
        """Test parse_llm_response with backticks but no json"""
        result = llm_chat.parse_llm_response("```not json```")
        assert result is None

    def test_parse_llm_response_empty(self, llm_chat):
        """Test parse_llm_response with empty input"""
        result = llm_chat.parse_llm_response("")
        assert result is None

    def test_process_error(self, llm_chat, mock_llm):
        """Test process propagates model errors"""
        mock_llm.process.side_effect = Exception("Model error")
        with pytest.raises(Exception, match="Model error"):
            llm_chat.process("input")
        llm_chat.logger.debug.assert_not_called()
