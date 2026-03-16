"""MLflow prompt registration helpers."""

from __future__ import annotations

from dataclasses import dataclass

import mlflow

from .config import Config
from .prompt import PROMPT_TEMPLATE


@dataclass(frozen=True)
class RegisteredPrompt:
    """Represents a prompt registered with MLflow."""

    name: str
    uri: str
    version: int


def register_prompt(config: Config, commit_message: str = "Register prompt") -> RegisteredPrompt:
    """Register or update the prompt in the MLflow tracking server."""

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    prompt = mlflow.genai.register_prompt(
        name=config.prompt_template_name,
        template=PROMPT_TEMPLATE,
        commit_message=commit_message,
    )

    return RegisteredPrompt(name=prompt.name, uri=prompt.uri, version=prompt.version)
