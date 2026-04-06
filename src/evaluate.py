"""Evaluation helpers for the ticket classifier demo."""

from __future__ import annotations

import logging

import mlflow

from .config import Config
from .predict import _get_llm_client, _get_prompt_template, predict_from_inputs
from .prompt import load_prompt_uri
from .scorers import exact_category_match

logger = logging.getLogger(__name__)


def _resolve_prompt_uri(prompt_uri: str) -> str:
    """Resolve a prompt URI alias (for example @latest) to a concrete version URI."""
    if "@" not in prompt_uri:
        return prompt_uri

    prompt_obj = mlflow.genai.load_prompt(prompt_uri)
    prompt_name = getattr(prompt_obj, "name", None)
    prompt_version = getattr(prompt_obj, "version", None)
    if prompt_name and prompt_version is not None:
        return f"prompts:/{prompt_name}/{prompt_version}"

    return prompt_uri


def evaluate(
    config: Config,
    data,
    prompt_uri: str | None = None,
    additional_scorers: list | None = None,
) -> dict:
    """Run a baseline evaluation of the current prompt + model.

    Args:
        config: Runtime configuration.
        data: Evaluation dataset (DataFrame or list of dicts with ``inputs``
            and ``expectations`` keys).
        prompt_uri: Optional explicit MLflow prompt URI, e.g.
            ``prompts:/support-ticket-classifier-prompt/1``.  When omitted the
            latest registered version is used.
        additional_scorers: Extra scorer functions to include alongside the
            default ``exact_category_match`` scorer.
    """
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    logger.debug("MLflow tracking URI: %s", config.mlflow_tracking_uri)

    requested_prompt_uri = prompt_uri or load_prompt_uri(config, version="latest")
    effective_prompt_uri = _resolve_prompt_uri(requested_prompt_uri)

    logger.info(
        "Starting evaluation with provider=%s, model=%s, prompt_uri=%s",
        config.llm_provider,
        config.model_name,
        effective_prompt_uri,
    )
    total_samples = len(data) if hasattr(data, "__len__") else None
    logger.debug(
        "Data size: %s samples",
        total_samples if total_samples is not None else "unknown",
    )

    mlflow.set_experiment(config.experiment_name)
    logger.debug("Experiment: %s", config.experiment_name)

    scorers = [exact_category_match]
    if additional_scorers:
        logger.debug("Adding %s custom scorers", len(additional_scorers))
        scorers.extend(additional_scorers)

    # End any lingering active run so this evaluation is fully isolated.
    mlflow.end_run()

    run_name = f"evaluate-{effective_prompt_uri}"
    logger.debug("Starting fresh MLflow run: %s", run_name)

    logger.debug("Preloading prompt template and LLM client for evaluation")
    prompt_template = _get_prompt_template(prompt_uri=effective_prompt_uri)
    llm_client = _get_llm_client(config)
    progress = {"completed": 0}

    def predict_with_progress(customer_message: str) -> str:
        """Reuse expensive evaluation dependencies across all samples."""
        result = predict_from_inputs(
            config,
            {"customer_message": customer_message},
            prompt_template=prompt_template,
            client=llm_client,
            traced=False,
        )
        progress["completed"] += 1
        if total_samples and (
            progress["completed"] == 1
            or progress["completed"] == total_samples
            or progress["completed"] % 10 == 0
        ):
            logger.info(
                "Evaluation progress: %s/%s samples completed",
                progress["completed"],
                total_samples,
            )
        return result

    # NOTE: The lambda parameter name *must* match the key in the `inputs` dict
    # of the evaluation dataset ("customer_message") so that MLflow can wire
    # the value through correctly.
    logger.debug("Running evaluation with %s scorers", len(scorers))
    with mlflow.start_run(run_name=run_name):
        result = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_with_progress,
            scorers=scorers,
        )

    logger.info("Evaluation completed successfully")
    logger.debug("Metrics: %s", result.metrics)
    return result.metrics


def print_metrics(metrics: dict) -> None:
    """Print evaluation metrics in a human-friendly format."""
    logger.info("Evaluation metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            logger.info("  %36s = %.4f", k, v)
        else:
            logger.info("  %36s = %s", k, v)
