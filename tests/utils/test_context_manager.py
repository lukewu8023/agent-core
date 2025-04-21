# tests/utils/test_context_manager.py

from agent_core.utils.context_manager import ContextManager


def test_add_remove_context():
    c = ContextManager()
    c.add_context("role", "user")
    assert "role" in c.context
    c.remove_context("role")
    assert "role" not in c.context


def test_context_to_str_empty():
    c = ContextManager()
    result = c.context_to_str()
    assert result == "", "Empty context should produce empty string"


def test_context_to_str_non_empty():
    c = ContextManager()
    c.add_context("foo", "bar")
    output = c.context_to_str()
    assert "*foo*" in output and "bar" in output
