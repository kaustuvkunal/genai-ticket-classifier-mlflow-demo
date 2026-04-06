"""Custom scoring functions used in evaluation and optimization."""

from __future__ import annotations

from typing import Any, Dict

from mlflow.genai import scorer


_VALID_CATEGORIES = ["incident", "request", "problem", "change"]


def _extract_category(text: str) -> str:
    """Extract one of the known categories from the model output.

    This is intentionally tolerant: we allow extra text but prioritize the first known
    category token found.
    """

    normalized = str(text or "").lower().strip()
    for category in _VALID_CATEGORIES:
        if category in normalized:
            return category
    return "unknown"


@scorer(name="exact_category_match")
def exact_category_match(outputs: Any, expectations: Dict[str, Any]) -> bool:
    """Return True if predicted category matches ground truth (case-insensitive)."""

    expected = str(expectations.get("type", "")).strip().lower()

    if isinstance(outputs, dict):
        predicted = str(outputs.get("type", "")).strip().lower()
    else:
        predicted = _extract_category(outputs)

    return predicted == expected
