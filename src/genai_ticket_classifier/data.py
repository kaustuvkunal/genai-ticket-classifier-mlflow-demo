"""Data loading utilities for the ticket classifier demo."""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset


def load_eval_data() -> pd.DataFrame:
    """Load the evaluation set as a pandas DataFrame.

    Returns:
        A DataFrame with columns `inputs` and `expectations`, which is compatible with
        `mlflow.genai.evaluate`.
    """

    dataset_dict = load_dataset("kaustuvkunal/support-message-categorization")
    gold_df = dataset_dict["gold_examples"].to_pandas()

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
