"""Configuration helpers for the ticket classifier demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    """Runtime configuration for the ticket classifier demo."""

    groq_api_key: str
    mlflow_tracking_uri: str
    model_name: str
    prompt_template_name: str
    experiment_name: str


def load_config(env_path: Optional[str] = None) -> Config:
    """Load configuration from environment variables.

    It loads environment variables from a `.env` file if present.
    """

    # Prefer an explicit env file, otherwise look in the repo root.
    env_path = env_path or os.environ.get("TICKET_CLASSIFIER_ENV_PATH")
    if env_path:
        load_dotenv(env_path)
    else:
        # Load from .env in the current working directory for convenience.
        candidate = Path.cwd() / ".env"
        if candidate.exists():
            load_dotenv(candidate)

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set. See .env.example for guidance.")

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_name = os.getenv("MODEL_NAME") or "llama-3.1-8b-instant"
    prompt_template_name = os.getenv("PROMPT_NAME") or "support-ticket-classifier-prompt"
    experiment_name = os.getenv("MLFLOW_EXPERIMENT") or "Support_Ticket_Classification_project"

    return Config(
        groq_api_key=groq_api_key,
        mlflow_tracking_uri=mlflow_tracking_uri,
        model_name=model_name,
        prompt_template_name=prompt_template_name,
        experiment_name=experiment_name,
    )
