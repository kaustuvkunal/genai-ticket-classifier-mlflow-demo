"""Evaluation helpers for the ticket classifier demo."""

from __future__ import annotations

from typing import List

import mlflow

from .config import Config
from .predict import predict, predict_from_inputs
from .scorers import exact_category_match


def evaluate(config: Config, data, additional_scorers: List = None) -> dict:
    """Run a baseline evaluation of the current prompt + model."""

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    scorers = [exact_category_match]
    if additional_scorers:
        scorers.extend(additional_scorers)

    result = mlflow.genai.evaluate(
        data=data,
        predict_fn=lambda x: predict_from_inputs(config, x),
        scorers=scorers,
    )

    return result.metrics


def print_metrics(metrics: dict) -> None:
    """Print evaluation metrics in a human-friendly format."""

    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k:36} = {v:.4f}")
        else:
            print(f"  {k:36} = {v}")
