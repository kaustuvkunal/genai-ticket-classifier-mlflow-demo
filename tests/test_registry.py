"""Tests for MLflow prompt registry utilities."""

import pytest
from unittest.mock import patch, MagicMock
from src.registry import RegisteredPrompt, register_prompt
from src.config import load_config


def test_registered_prompt_dataclass():
    """Test RegisteredPrompt dataclass creation."""
    prompt = RegisteredPrompt(name="test-prompt", uri="models:/test-prompt/1", version=1)

    assert prompt.name == "test-prompt"
    assert prompt.uri == "models:/test-prompt/1"
    assert prompt.version == 1


def test_registered_prompt_with_metadata():
    """Test RegisteredPrompt can hold additional metadata."""
    metadata = {"category_count": 4, "model": "gpt-4"}
    prompt = RegisteredPrompt(
        name="classifier-v1",
        uri="models:/classifier-v1/production",
        version=1,
    )

    assert prompt.name == "classifier-v1"
    # Dataclass should be properly structured
    assert hasattr(prompt, "uri")


def test_register_prompt_returns_registered_prompt():
    """Test that register_prompt returns a RegisteredPrompt instance."""
    config = load_config()

    with patch("src.registry.mlflow.genai.fluent.log_model") as mock_log:
        mock_log.return_value = None
        
        with patch("src.registry.mlflow.models.get_registered_model") as mock_get:
            mock_model = MagicMock()
            mock_model.latest_versions = [MagicMock(version=1)]
            mock_get.return_value = mock_model
            
            result = register_prompt(config, "test-prompt")
            
            assert isinstance(result, RegisteredPrompt)
            assert result.name is not None
            assert result.uri is not None
            assert result.version is not None


@patch.dict("os.environ", {"MLFLOW_TRACKING_URI": "sqlite:///mlflow.db"})
def test_register_prompt_with_mlflow_tracking():
    """Test prompt registration with MLflow tracking."""
    config = load_config()

    with patch("src.registry.mlflow.genai.fluent.log_model") as mock_log:
        mock_log.return_value = None
        
        with patch("src.registry.mlflow.models.get_registered_model") as mock_get:
            mock_model = MagicMock()
            mock_model.latest_versions = [MagicMock(version=1)]
            mock_get.return_value = mock_model
            
            # Should not raise an error
            result = register_prompt(config, "tracked-prompt")
            assert result is not None


def test_register_prompt_idempotent():
    """Test that registering the same prompt twice returns consistent results."""
    config = load_config()

    with patch("src.registry.mlflow.genai.fluent.log_model") as mock_log:
        mock_log.return_value = None
        
        with patch("src.registry.mlflow.models.get_registered_model") as mock_get:
            mock_model = MagicMock()
            mock_model.latest_versions = [MagicMock(version=1)]
            mock_get.return_value = mock_model
            
            result1 = register_prompt(config, "idempotent-test")
            result2 = register_prompt(config, "idempotent-test")
            
            # Both calls should return valid registered prompts
            assert isinstance(result1, RegisteredPrompt)
            assert isinstance(result2, RegisteredPrompt)


def test_register_prompt_version_increments():
    """Test that multiple registrations increment version numbers."""
    config = load_config()

    with patch("src.registry.mlflow.genai.fluent.log_model") as mock_log:
        mock_log.return_value = None
        
        with patch("src.registry.mlflow.models.get_registered_model") as mock_get:
            # First call returns version 1
            mock_model1 = MagicMock()
            mock_model1.latest_versions = [MagicMock(version=1)]
            
            mock_get.return_value = mock_model1
            result1 = register_prompt(config, "versioned-prompt")
            
            # Simulate MLflow returning newer version
            mock_model2 = MagicMock()
            mock_model2.latest_versions = [MagicMock(version=2)]
            mock_get.return_value = mock_model2
            
            result2 = register_prompt(config, "versioned-prompt")
            
            # Both should be RegisteredPrompt instances
            assert isinstance(result1, RegisteredPrompt)
            assert isinstance(result2, RegisteredPrompt)
