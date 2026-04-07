"""Prompt optimization helpers for the ticket classifier demo."""

from __future__ import annotations

import logging

import mlflow

from mlflow.genai.optimize.optimizers import GepaPromptOptimizer

from .config import Config
from .scorers import exact_category_match
from .prompt import load_prompt_uri
from .predict import predict_from_inputs

logger = logging.getLogger(__name__)


def optimize_prompt(
    config: Config,
    train_data,
    prompt_uri: str | None = None,
    prompt_version: str = "latest",
    max_metric_calls: int | None = None,
    display_progress_bar: bool = True,
) -> str:
    """Optimize the registered prompt via MLflow GenAI prompt optimization.

    Args:
        config: Runtime configuration.
        train_data: Training dataset.
        prompt_uri: Explicit MLflow prompt URI, e.g.
            ``prompts:/support-ticket-classifier-prompt/1``.  When provided,
            ``prompt_version`` is ignored.
        prompt_version: Version alias used when ``prompt_uri`` is not given
            (default: ``"latest"``).
        max_metric_calls: Override for the maximum scorer calls.  When
            ``None``, the value is taken from ``config.max_metric_calls``
            (i.e. ``MAX_METRIC_CALLS`` in ``.env``).
        display_progress_bar: Whether to show a progress bar.

    Returns:
        The URI of the optimized prompt.
    """
    effective_max_calls = max_metric_calls if max_metric_calls is not None else config.max_metric_calls
    logger.info(
        "Starting prompt optimization with provider=%s, max_metric_calls=%s",
        config.llm_provider,
        effective_max_calls,
    )
    logger.debug(
        "Prompt version: %s, progress_bar: %s",
        prompt_version,
        display_progress_bar,
    )

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    logger.debug("MLflow tracking URI: %s", config.mlflow_tracking_uri)

    mlflow.set_experiment(config.experiment_name)
    logger.debug("Experiment: %s", config.experiment_name)

    # Reflection model is configured per-provider in GROQ_REFLECTION_MODEL /
    # OPENAI_REFLECTION_MODEL env vars (loaded into config.reflection_model).
    reflection_model = config.reflection_model
    logger.debug("Reflection model: %s", reflection_model)

    optimizer = GepaPromptOptimizer(
        reflection_model=reflection_model,
        max_metric_calls=effective_max_calls,
        display_progress_bar=display_progress_bar,
    )

    prompt_uri = prompt_uri or load_prompt_uri(config, version=prompt_version)
    logger.debug("Input prompt URI: %s", prompt_uri)

    # End any lingering active run so this optimization is fully isolated.
    mlflow.end_run()

    run_name = f"optimize-{prompt_uri}"
    logger.debug("Starting fresh MLflow run: %s", run_name)
    logger.debug("Starting optimization process")
    with mlflow.start_run(run_name=run_name):
        opt_result = mlflow.genai.optimize_prompts(
            predict_fn=lambda customer_message: predict_from_inputs(
                config, {"customer_message": customer_message}, prompt_uri=prompt_uri
            ),
            train_data=train_data,
            prompt_uris=[prompt_uri],
            optimizer=optimizer,
            scorers=[exact_category_match],
        )

    optimized_prompt = opt_result.optimized_prompts[0]
    original_prompt = mlflow.genai.load_prompt(prompt_uri)

    original_template = (getattr(original_prompt, "template", "") or "").strip()
    optimized_template = (getattr(optimized_prompt, "template", "") or "").strip()

    if original_template == optimized_template:
        logger.warning(
            "Optimization produced no prompt template changes. "
            "MLflow may still register a new version even when content is unchanged."
        )

    run_id = getattr(opt_result, "run_id", None)
    if run_id:
        run_metrics = mlflow.get_run(run_id).data.metrics
        initial_score = run_metrics.get("initial_eval_score.accuracy")
        final_score = run_metrics.get("final_eval_score.accuracy")
        if initial_score is not None and final_score is not None:
            logger.info(
                "Optimization accuracy delta (final - initial): %.6f",
                final_score - initial_score,
            )
            if final_score <= initial_score:
                logger.warning(
                    "No accuracy improvement detected from prompt optimization "
                    "(initial=%.6f, final=%.6f).",
                    initial_score,
                    final_score,
                )

    logger.info("Optimization completed successfully")
    logger.info("Optimized prompt URI: %s", optimized_prompt.uri)

    return optimized_prompt.uri
