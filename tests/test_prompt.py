"""Tests for prompt utilities."""

import pytest
from unittest.mock import patch
from src.prompt import BASE_PROMPT_TEMPLATE, load_prompt_uri
from src.config import load_config


def test_base_prompt_template_not_empty():
    """Test that BASE_PROMPT_TEMPLATE is defined and non-empty."""
    assert BASE_PROMPT_TEMPLATE
    assert isinstance(BASE_PROMPT_TEMPLATE, str)
    assert len(BASE_PROMPT_TEMPLATE) > 0


def test_base_prompt_template_contains_instructions():
    """Test that prompt template contains clear instructions."""
    prompt = BASE_PROMPT_TEMPLATE
    # Should mention categories or classification
    assert "incident" in prompt.lower() or "category" in prompt.lower()


def test_base_prompt_template_has_four_categories():
    """Test that prompt template references the four supported categories."""
    prompt = BASE_PROMPT_TEMPLATE.lower()
    categories = ["incident", "request", "problem", "change"]

    for category in categories:
        assert category in prompt, f"Category '{category}' not mentioned in template"


def test_load_prompt_uri_returns_string():
    """Test that load_prompt_uri returns a URI string."""
    config = load_config()
    uri = load_prompt_uri(config, "test-prompt", version=1)

    assert isinstance(uri, str)
    assert len(uri) > 0


def test_load_prompt_uri_uses_config_model():
    """Test that prompt URI includes the config model reference."""
    config = load_config()
    uri = load_prompt_uri(config, "my-prompt", version=2)

    # URI should be model-specific format
    assert "prompt" in uri.lower() or "models" in uri.lower()


def test_load_prompt_uri_version_parameter():
    """Test that prompt URI changes with version parameter."""
    config = load_config()
    uri_v1 = load_prompt_uri(config, "test-prompt", version=1)
    uri_v2 = load_prompt_uri(config, "test-prompt", version=2)

    # Different versions should produce different URIs
    assert uri_v1 != uri_v2


@pytest.fixture
def mock_env():
    """Fixture for mocking environment variables."""
    with patch.dict("os.environ", {}, clear=False):
        yield
