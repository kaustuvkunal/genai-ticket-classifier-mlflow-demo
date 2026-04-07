"""Command line interface for the GenAI ticket classifier demo.

Usage:
    python -m src.cli --help
    python -m src.cli register-prompt
    python -m src.cli evaluate --limit 10
    python -m src.cli predict "My login is not working"
"""

from __future__ import annotations

import logging
import sys
from typing import Callable, TypeVar

import click

from .config import load_config
from .data import load_eval_data
from .evaluate import evaluate, print_metrics
from .optimize import optimize_prompt
from .predict import predict as predict_ticket
from .registry import register_prompt

logger = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def _configure_logging() -> None:
    """Configure logging once for all CLI commands."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def _run_or_exit(command_name: str, action: Callable[..., T], *args, **kwargs) -> T:
    """Run a command action and convert unexpected exceptions into clean CLI exits."""
    try:
        return action(*args, **kwargs)
    except (click.ClickException, RuntimeError, ValueError, OSError, KeyError) as error:
        _exit_with_error(command_name, error)


def _exit_with_error(command_name: str, error: Exception) -> None:
    """Print a consistent error message and exit the CLI command."""
    logger.error("%s failed: %s: %s", command_name, type(error).__name__, error)
    click.echo(f"Error: {error}", err=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Command implementations (business logic)
# ---------------------------------------------------------------------------
def _register_prompt(commit_message: str) -> None:
    """Register the prompt and print resulting metadata."""
    config = load_config()
    logger.debug("Config loaded: provider=%s", config.llm_provider)

    prompt = register_prompt(config, commit_message=commit_message)
    click.echo(f"Registered prompt: {prompt.name} (version {prompt.version})")
    click.echo(f"Prompt URI: {prompt.uri}")
    logger.info("register-prompt completed successfully")


def _evaluate(prompt_uri: str | None, skip_data: bool, limit: int | None) -> None:
    """Run evaluation and print metric summary."""
    config = load_config()
    logger.debug("Config loaded: provider=%s", config.llm_provider)

    if skip_data:
        click.echo("Skipping data load (no evaluation data).")
        logger.info("Evaluation skipped per user request")
        sys.exit(0)

    logger.debug("Loading evaluation data")
    data = load_eval_data(limit=limit)
    logger.debug("Evaluation data loaded: %s samples", len(data))

    logger.debug("Starting evaluation")
    metrics = evaluate(config=config, data=data, prompt_uri=prompt_uri)
    print_metrics(metrics)
    logger.info("evaluate completed successfully")


def _optimize(prompt_uri: str | None, prompt_version: str, max_metric_calls: int) -> None:
    """Run prompt optimization and print optimized prompt URI."""
    config = load_config()
    logger.debug("Config loaded: provider=%s", config.llm_provider)

    logger.debug("Loading training data")
    data = load_eval_data()
    logger.debug("Training data loaded: %s samples", len(data))

    logger.debug("Starting prompt optimization")
    optimized_uri = optimize_prompt(
        config=config,
        train_data=data,
        prompt_uri=prompt_uri,
        prompt_version=prompt_version,
        max_metric_calls=max_metric_calls,
    )

    click.echo(f"Optimized prompt URI: {optimized_uri}")
    logger.info("optimize completed successfully")


def _predict(message: tuple[str, ...], prompt_uri: str | None) -> None:
    """Run a single-message prediction and print the label."""
    if not message:
        logger.error("No message provided")
        raise click.UsageError("Provide a customer message to classify.")

    customer_message = " ".join(message).strip()
    logger.debug("Message to predict: %s chars", len(customer_message))

    config = load_config()
    logger.debug("Config loaded: provider=%s", config.llm_provider)

    if prompt_uri:
        click.echo(f"Prompt : {prompt_uri}")
    else:
        click.echo("Prompt : src/prompt.py (local fallback)")

    logger.debug("Starting prediction")
    prediction = predict_ticket(config, customer_message, prompt_uri)  # pylint: disable=redundant-keyword-arg

    click.echo(f"Result : {prediction}")
    logger.info("Prediction: %s", prediction)


@click.group()
def main() -> None:
    """Run GenAI support-ticket CLI commands.

    Tip: run `python -m src.cli COMMAND --help` for command-specific usage.
    """
    _configure_logging()
    logger.debug("CLI initialized")


# ---------------------------------------------------------------------------
# Click command handlers (all *_cmd functions in one block)
# ---------------------------------------------------------------------------
@main.command("register-prompt")
@click.option(
    "--commit-message",
    default="Register prompt",
    show_default=True,
    help="Commit message to associate with the prompt registration.",
)
def register_prompt_cmd(commit_message: str) -> None:
    """Register or update the base prompt in the MLflow tracking server."""
    logger.info("Executing register-prompt command")
    _run_or_exit("register-prompt", _register_prompt, commit_message)


@main.command("evaluate")
@click.option(
    "--prompt-uri",
    default=None,
    show_default=True,
    help="MLflow prompt URI to evaluate, e.g. prompts:/support-ticket-classifier-prompt/1. "
         "Defaults to the latest registered version.",
)
@click.option(
    "--skip-data",
    is_flag=True,
    default=False,
    help="Skip loading the canonical evaluation dataset (useful for debugging).",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    show_default=True,
    help="Evaluate only the first N samples. Useful during development to get faster feedback.",
)
def evaluate_cmd(prompt_uri: str, skip_data: bool, limit: int | None) -> None:
    """Run baseline evaluation on the canonical evaluation dataset."""
    logger.info("Executing evaluate command")
    if prompt_uri:
        logger.info("Using prompt URI: %s", prompt_uri)
    _run_or_exit("evaluate", _evaluate, prompt_uri, skip_data, limit)


@main.command("optimize")
@click.option(
    "--prompt-uri",
    default=None,
    show_default=True,
    help="MLflow prompt URI to optimize, e.g. prompts:/support-ticket-classifier-prompt/1. "
         "When provided, --prompt-version is ignored.",
)
@click.option(
    "--prompt-version",
    default="latest",
    show_default=True,
    help="The prompt version to optimize (used when --prompt-uri is not given).",
)
@click.option(
    "--max-metric-calls",
    default=None,
    type=int,
    show_default=True,
    help="Maximum scorer calls during optimization. "
         "Defaults to MAX_METRIC_CALLS from .env (currently 64).",
)
def optimize_cmd(prompt_uri: str | None, prompt_version: str, max_metric_calls: int | None) -> None:
    """Optimize the registered prompt using MLflow GenAI optimization."""
    logger.info("Executing optimize command")
    logger.debug(
        "Options: prompt_version=%s, max_metric_calls=%s",
        prompt_version,
        max_metric_calls,
    )
    _run_or_exit("optimize", _optimize, prompt_uri, prompt_version, max_metric_calls)


@main.command("predict")
@click.argument("message", nargs=-1)
@click.option(
    "--prompt-uri",
    default=None,
    show_default=True,
    help="MLflow prompt URI to use, e.g. prompts:/support-ticket-classifier-prompt/1. "
         "Defaults to the latest registered version, falling back to prompts/finalise_prompt.py.",
)
def predict_cmd(message: tuple[str, ...], prompt_uri: str | None) -> None:
    """Predict the category for a single customer message."""
    logger.info("Executing predict command")
    _run_or_exit("predict", _predict, message, prompt_uri)


# Backward-compatible alias used by tests/importers.
cli = main


if __name__ == "__main__":
    main()
