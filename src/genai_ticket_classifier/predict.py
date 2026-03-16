"""Prediction logic for the ticket classifier demo."""

from __future__ import annotations

from typing import Any

import mlflow
from groq import Groq

from .config import Config
from .prompt import load_prompt_uri


def _build_messages(prompt_template: str, customer_message: str) -> list[dict[str, str]]:
    """Build the chat message history for the Groq chat completion API."""
    return [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": customer_message},
    ]


@mlflow.trace(name="ticket_classifier", span_type="LLM")
def predict(config: Config, customer_message: str) -> str:
    """Predict the ticket category for a single customer message."""

    # NOTE: Groq client reads the API key from the environment.
    client = Groq()

    prompt_uri = load_prompt_uri(config)
    prompt_obj = mlflow.genai.load_prompt(prompt_uri)

    messages = _build_messages(prompt_obj.template, customer_message)

    resp = client.chat.completions.create(
        model=config.model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=32,
    )

    return resp.choices[0].message.content.strip()


def predict_batch(config: Config, inputs: list[str]) -> list[str]:
    """Predict categories for a batch of messages."""
    return [predict(config, text) for text in inputs]


def predict_from_inputs(config: Config, inputs: dict[str, str]) -> str:
    """Wrapper for MLflow GenAI evaluation/optimization.

    MLflow expects `predict_fn` to accept the `inputs` value from the evaluation dataset.
    """

    customer_message = inputs.get("customer_message")
    if customer_message is None:
        raise ValueError("Missing 'customer_message' in inputs")

    return predict(config, customer_message)
