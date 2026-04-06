"""MLflow prompt registration helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlflow

from .config import Config
from . import prompt as prompt_module

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredPrompt:
    """Represents a prompt registered with MLflow."""

    name: str
    uri: str
    version: int


def register_prompt(config: Config, commit_message: str = "Register prompt") -> RegisteredPrompt:
    """Register or update the prompt in the MLflow tracking server."""
    logger.info("Registering prompt: %s", config.prompt_template_name)
    logger.debug("Commit message: %s", commit_message)
    logger.debug("MLflow tracking URI: %s", config.mlflow_tracking_uri)

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    logger.debug("Set MLflow tracking URI")

    mlflow.set_experiment(config.experiment_name)
    logger.debug("Set experiment: %s", config.experiment_name)

    logger.debug("Registering prompt with template size: %s chars", len(prompt_module.BASE_PROMPT_TEMPLATE))
    prompt = mlflow.genai.register_prompt(
        name=config.prompt_template_name,
        template=prompt_module.BASE_PROMPT_TEMPLATE,
        commit_message=commit_message,
    )

    logger.info("Prompt registered successfully: %s (version %s)", prompt.name, prompt.version)
    logger.debug("Prompt URI: %s", prompt.uri)

    return RegisteredPrompt(name=prompt.name, uri=prompt.uri, version=prompt.version)
