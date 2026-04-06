"""Data loading utilities for the ticket classifier demo."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Path to the local gold evaluation CSV relative to this file's package root.
_LOCAL_GOLD_CSV = Path(__file__).parent.parent / "dataset" / "support_ticket_categories_gold_examples.csv"


def load_eval_data(limit: int | None = None) -> pd.DataFrame:
    """Load the evaluation set as a pandas DataFrame.

    Reads from the local ``dataset/support_ticket_categories_gold_examples.csv``
    file — no network download required.  Pass ``limit`` to cap the number of
    rows evaluated (useful during development to get faster feedback).

    Args:
        limit: If set, only the first *limit* rows are returned.

    Returns:
        A DataFrame with columns ``inputs`` and ``expectations``, compatible
        with ``mlflow.genai.evaluate``.
    """
    logger.debug("Loading evaluation data from %s", _LOCAL_GOLD_CSV)
    gold_df = pd.read_csv(_LOCAL_GOLD_CSV)
    if limit is not None:
        gold_df = gold_df.head(limit)
        logger.info("Limiting evaluation to %s samples (--limit %s)", len(gold_df), limit)
    else:
        logger.info("Loaded %s evaluation samples from local CSV", len(gold_df))

    eval_df = pd.DataFrame(
        [
            {
                "inputs": {"customer_message": row["customer_message"]},
                "expectations": {"type": row["type"]},
            }
            for _, row in gold_df.iterrows()
        ]
    )

    return eval_df
