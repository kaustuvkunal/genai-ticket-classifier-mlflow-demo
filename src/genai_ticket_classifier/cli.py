"""Command line interface for the GenAI ticket classifier demo."""

from __future__ import annotations

import sys

import click

from .config import load_config
from .data import load_eval_data
from .evaluate import evaluate, print_metrics
from .optimize import optimize_prompt
from .registry import register_prompt
from .predict import predict


@click.group()
def main() -> None:
    """GenAI ticket classifier demo CLI."""


@main.command("register-prompt")
@click.option(
    "--commit-message",
    default="Register prompt",
    show_default=True,
    help="Commit message to associate with the prompt registration.",
)
def register_prompt_cmd(commit_message: str) -> None:
    """Register or update the base prompt in the MLflow tracking server."""

    config = load_config()
    prompt = register_prompt(config, commit_message=commit_message)
    click.echo(f"Registered prompt: {prompt.name} (version {prompt.version})")
    click.echo(f"Prompt URI: {prompt.uri}")


@main.command("evaluate")
@click.option(
    "--skip-data",
    is_flag=True,
    default=False,
    help="Skip loading the canonical evaluation dataset (useful for debugging).",
)
def evaluate_cmd(skip_data: bool) -> None:
    """Run baseline evaluation on the canonical evaluation dataset."""

    config = load_config()

    if skip_data:
        click.echo("Skipping data load (no evaluation data).")
        sys.exit(0)

    data = load_eval_data()
    metrics = evaluate(config=config, data=data)
    print_metrics(metrics)


@main.command("optimize")
@click.option(
    "--prompt-version",
    default="latest",
    show_default=True,
    help="The prompt version to optimize.",
)
@click.option(
    "--max-metric-calls",
    default=400,
    show_default=True,
    help="Maximum number of scorer calls to make during optimization.",
)
def optimize_cmd(prompt_version: str, max_metric_calls: int) -> None:
    """Optimize the registered prompt using MLflow GenAI optimization."""

    config = load_config()
    data = load_eval_data()

    optimized_uri = optimize_prompt(
        config=config,
        train_data=data,
        prompt_version=prompt_version,
        max_metric_calls=max_metric_calls,
    )

    click.echo(f"Optimized prompt URI: {optimized_uri}")


@main.command("predict")
@click.argument("message", nargs=-1)
def predict_cmd(message: tuple[str, ...]) -> None:
    """Predict the category for a single customer message."""

    if not message:
        raise click.UsageError("Provide a customer message to classify.")

    customer_message = " ".join(message).strip()
    config = load_config()

    prediction = predict(config, customer_message)

    click.echo(prediction)
