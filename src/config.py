"""Configuration helpers for the ticket classifier demo."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = {"groq", "openai"}


@dataclass(frozen=True)
class Config:
    """Runtime configuration for the ticket classifier demo.

    All values are sourced from environment variables (via .env).  API keys
    are intentionally excluded to prevent accidental exposure via MLflow
    traces, logging, or repr serialization — they are read at the point of
    use inside the LLM client factory.
    """

    # Provider / model identity
    llm_provider: str
    model_name: str
    reflection_model: str       # stronger model used by the optimizer

    # MLflow settings
    mlflow_tracking_uri: str
    experiment_name: str
    prompt_template_name: str

    # Optimization settings
    max_metric_calls: int       # MAX_METRIC_CALLS

    # Inference settings
    temperature: float          # LLM_TEMPERATURE
    max_tokens: int             # LLM_MAX_TOKENS


def load_config(env_path: str | None = None) -> Config:
    """Load configuration from environment variables.

    Loads environment variables from an explicit path or from ``.env`` in the
    current working directory when available.
    """
    logger.debug("Loading configuration from env_path=%s", env_path)

    # Prefer an explicit env file, otherwise look in the repo root.
    env_path = env_path or os.environ.get("TICKET_CLASSIFIER_ENV_PATH")
    if env_path:
        logger.info("Loading environment from explicit path: %s", env_path)
        load_dotenv(env_path)
    else:
        candidate = Path.cwd() / ".env"
        if candidate.exists():
            logger.info("Loading environment from: %s", candidate)
            load_dotenv(candidate)
        else:
            logger.debug("No .env file found in current directory")

    llm_provider = os.getenv("LLM_PROVIDER", "groq").lower()
    logger.info("LLM provider configured: %s", llm_provider)

    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if llm_provider == "groq" and not groq_api_key:
        logger.error("GROQ_API_KEY is not set but Groq provider is selected")
        raise RuntimeError(
            "GROQ_API_KEY is not set. Please set it in .env file or environment variables. "
            "See .env.example for guidance."
        )
    if llm_provider == "openai" and not openai_api_key:
        logger.error("OPENAI_API_KEY is not set but OpenAI provider is selected")
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please set it in .env file or environment variables. "
            "See .env.example for guidance."
        )
    if llm_provider not in SUPPORTED_PROVIDERS:
        logger.error("Unsupported LLM provider: %s", llm_provider)
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER: {llm_provider}. "
            "Supported providers: groq, openai"
        )

    # Model name: generic MODEL_NAME takes priority, then provider-specific vars.
    if llm_provider == "groq":
        model_name = os.getenv("MODEL_NAME") or os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
        _reflection_base = os.getenv("GROQ_REFLECTION_MODEL", "llama-3.3-70b-versatile")
        reflection_model = f"groq:/{_reflection_base}"
    else:  # openai
        model_name = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        _reflection_base = os.getenv("OPENAI_REFLECTION_MODEL", "gpt-4o")
        reflection_model = f"openai:/{_reflection_base}"

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "Support_Ticket_Classification_project")
    prompt_template_name = os.getenv("PROMPT_NAME", "support-ticket-classifier-prompt")

    max_metric_calls = int(os.getenv("MAX_METRIC_CALLS", "64"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "32"))

    logger.info("Configuration loaded successfully")
    logger.debug(
        "Config: provider=%s, model=%s, prompt_name=%s, experiment=%s, "
        "max_metric_calls=%s, temperature=%s, max_tokens=%s",
        llm_provider, model_name, prompt_template_name, experiment_name,
        max_metric_calls, temperature, max_tokens,
    )

    # API keys are NOT stored in Config — they are read from environment
    # at the point of use inside _get_llm_client() to prevent leakage via
    # MLflow traces, logging, or any other serialization path.
    return Config(
        llm_provider=llm_provider,
        model_name=model_name,
        reflection_model=reflection_model,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        prompt_template_name=prompt_template_name,
        max_metric_calls=max_metric_calls,
        temperature=temperature,
        max_tokens=max_tokens,
    )
