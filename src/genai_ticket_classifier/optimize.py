"""Prompt optimization helpers for the ticket classifier demo."""

from __future__ import annotations

from typing import Optional

import mlflow

from mlflow.genai.optimize.optimizers import GepaPromptOptimizer

from .config import Config
from .scorers import exact_category_match
from .prompt import load_prompt_uri
from .predict import predict_from_inputs


def optimize_prompt(
    config: Config,
    train_data,
    prompt_version: str = "latest",
    max_metric_calls: int = 400,
    display_progress_bar: bool = True,
) -> str:
    """Optimize the registered prompt via MLflow GenAI prompt optimization.

    Returns:
        The URI of the optimized prompt.
    """

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    optimizer = GepaPromptOptimizer(
        reflection_model="groq:/llama-3.3-70b-versatile",
        max_metric_calls=max_metric_calls,
        display_progress_bar=display_progress_bar,
    )

    prompt_uri = load_prompt_uri(config, version=prompt_version)

    opt_result = mlflow.genai.optimize_prompts(
        predict_fn=lambda x: predict_from_inputs(config, x),
        train_data=train_data,
        prompt_uris=[prompt_uri],
        optimizer=optimizer,
        scorers=[exact_category_match],
    )

    optimized_prompt = opt_result.optimized_prompts[0]
    return optimized_prompt.uri
