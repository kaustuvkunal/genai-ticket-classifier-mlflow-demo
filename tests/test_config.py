import os

import pytest

from src.config import Config, load_config


def test_load_config_missing_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text("# empty")

    with pytest.raises(RuntimeError):
        load_config(env_path=str(env_file))


def test_load_config_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MODEL_NAME", "llama-3.1-8b-instant")

    cfg = load_config()

    assert isinstance(cfg, Config)
    assert os.environ.get("GROQ_API_KEY") == "test-key"
    assert cfg.mlflow_tracking_uri == "http://localhost:5000"
    assert cfg.model_name == "llama-3.1-8b-instant"


def test_load_config_openai_provider(monkeypatch):
    """Test OpenAI provider selection and validation."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    cfg = load_config()

    assert cfg.llm_provider == "openai"
    assert cfg.model_name == "gpt-4"  # Default model for OpenAI


def test_load_config_openai_missing_key(monkeypatch, tmp_path):
    """Test that OpenAI provider fails without API key."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text("# empty")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        load_config(env_path=str(env_file))


def test_load_config_unsupported_provider(monkeypatch, tmp_path):
    """Test that unsupported provider raises error."""
    monkeypatch.setenv("LLM_PROVIDER", "unsupported-provider")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text("# empty")

    with pytest.raises(RuntimeError, match="Unsupported LLM_PROVIDER"):
        load_config(env_path=str(env_file))


def test_load_config_custom_model_name(monkeypatch):
    """Test that custom MODEL_NAME overrides defaults."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_NAME", "custom-model-name")

    cfg = load_config()

    assert cfg.model_name == "custom-model-name"


def test_load_config_defaults(monkeypatch):
    """Test default values when env vars are not set."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("PROMPT_NAME", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT", raising=False)

    cfg = load_config()

    assert cfg.prompt_template_name == "support-ticket-classifier-prompt"
    assert cfg.experiment_name == "Support_Ticket_Classification_project"
    assert cfg.llm_provider == "groq"  # Default provider


def test_config_api_keys_not_stored(monkeypatch):
    """Verify API keys are never stored in Config object."""
    monkeypatch.setenv("GROQ_API_KEY", "sensitive-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sensitive-openai-key")

    cfg = load_config()

    # Config should not have groq_api_key or openai_api_key attributes
    assert not hasattr(cfg, "groq_api_key")
    assert not hasattr(cfg, "openai_api_key")
    # But environment should still have them
    assert os.environ.get("GROQ_API_KEY") == "sensitive-key"
