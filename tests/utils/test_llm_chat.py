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
