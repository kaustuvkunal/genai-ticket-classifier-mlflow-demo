"""Configuration helpers for the ticket classifier demo."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = {"groq", "openai"}
DEFAULT_MODEL_BY_PROVIDER = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4",
}


@dataclass(frozen=True)
class Config:
    """Runtime configuration for the ticket classifier demo.

    API keys are intentionally excluded to prevent accidental exposure via
    MLflow traces, logging, or repr serialization.  They are read directly
    from environment variables at the point of use.
    """

    llm_provider: str
    mlflow_tracking_uri: str
    model_name: str
    prompt_template_name: str
    experiment_name: str


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

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_name = os.getenv("MODEL_NAME") or DEFAULT_MODEL_BY_PROVIDER.get(llm_provider, "")

    prompt_template_name = os.getenv("PROMPT_NAME") or "support-ticket-classifier-prompt"
    experiment_name = os.getenv("MLFLOW_EXPERIMENT") or "Support_Ticket_Classification_project"

    logger.info("Configuration loaded successfully")
    logger.debug(
        "Config: provider=%s, model=%s, prompt_name=%s, experiment=%s",
        llm_provider,
        model_name,
        prompt_template_name,
        experiment_name,
    )

    # API keys are NOT stored in Config — they are read from environment
    # at the point of use inside _get_llm_client() to prevent leakage via
    # MLflow traces, logging, or any other serialization path.
    return Config(
        llm_provider=llm_provider,
        mlflow_tracking_uri=mlflow_tracking_uri,
        model_name=model_name,
        prompt_template_name=prompt_template_name,
        experiment_name=experiment_name,
    )
