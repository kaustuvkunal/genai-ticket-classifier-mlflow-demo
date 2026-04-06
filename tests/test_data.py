"""Tests for data loading utilities."""

import pytest
import pandas as pd
from pathlib import Path

from src.data import load_eval_data


def test_load_eval_data_returns_dataframe():
    """Test that load_eval_data returns a DataFrame."""
    df = load_eval_data()

    assert isinstance(df, pd.DataFrame)
    assert "inputs" in df.columns
    assert "expectations" in df.columns


def test_load_eval_data_has_required_structure():
    """Test that loaded data has the correct structure."""
    df = load_eval_data(limit=1)

    assert len(df) > 0
    # Each row should have inputs and expectations
    row = df.iloc[0]
    assert isinstance(row["inputs"], dict)
    assert isinstance(row["expectations"], dict)
    assert "customer_message" in row["inputs"]
    assert "type" in row["expectations"]


def test_load_eval_data_with_limit():
    """Test that limit parameter restricts the number of rows."""
    df_limited = load_eval_data(limit=5)
    df_all = load_eval_data(limit=None)

    assert len(df_limited) <= 5
    assert len(df_all) >= len(df_limited)


def test_load_eval_data_limit_none():
    """Test that limit=None loads all data."""
    df = load_eval_data(limit=None)

    assert len(df) > 0
    # All rows should be loaded (the actual number depends on the CSV)
    assert len(df) >= 1


def test_load_eval_data_large_limit():
    """Test that limit larger than file size returns all data."""
    df_limited = load_eval_data(limit=999999)
    df_all = load_eval_data(limit=None)

    # Both should be the same length
    assert len(df_limited) == len(df_all)


def test_load_eval_data_valid_categories():
    """Test that loaded data contains valid categories."""
    df = load_eval_data(limit=10)
    valid_categories = {"incident", "request", "problem", "change"}

    for _, row in df.iterrows():
        category = row["expectations"].get("type", "").lower()
        assert category in valid_categories, f"Invalid category: {category}"


def test_load_eval_data_no_empty_messages():
    """Test that all customer messages are non-empty."""
    df = load_eval_data(limit=10)

    for _, row in df.iterrows():
        message = row["inputs"].get("customer_message", "")
        assert message and len(message.strip()) > 0, "Found empty customer message"
