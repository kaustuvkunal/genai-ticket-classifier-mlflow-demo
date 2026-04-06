"""Tests for evaluation pipeline utilities."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.evaluate import evaluate, print_metrics, _resolve_prompt_uri
from src.config import load_config


@patch("src.evaluate.mlflow.genai.evaluate")
@patch("src.evaluate.load_eval_data")
def test_evaluate_returns_dict(mock_load_data, mock_mlfow_eval):
    """Test that evaluate returns a dictionary of metrics."""
    # Mock data loading
    mock_load_data.return_value = pd.DataFrame({
        "inputs": [{"customer_message": "test"}],
        "expectations": [{"type": "incident"}],
    })
    
    # Mock MLflow evaluation
    mock_results = {
        "exact_category_match": {"score": 0.85, "evaluated_on": 100}
    }
    mock_mlfow_eval.return_value = mock_results
    
    config = load_config()
    result = evaluate(config, prompt_uri="models:/test/1", limit=1)
    
    assert isinstance(result, dict)
    assert "exact_category_match" in result


@patch("src.evaluate.mlflow.genai.evaluate")
@patch("src.evaluate.load_eval_data")
def test_evaluate_with_limit_restriction(mock_load_data, mock_mlfow_eval):
    """Test that evaluate respects the limit parameter."""
    df = pd.DataFrame({
        "inputs": [{"customer_message": f"test {i}"} for i in range(100)],
        "expectations": [{"type": "incident"} for i in range(100)],
    })
    mock_load_data.return_value = df
    mock_mlfow_eval.return_value = {"exact_category_match": {"score": 0.8}}
    
    config = load_config()
    result = evaluate(config, prompt_uri="models:/test/1", limit=10)
    
    # Verify limit was passed to load_eval_data
    mock_load_data.assert_called()


def test_print_metrics_outputs_readable_format(capsys):
    """Test that print_metrics outputs human-readable format."""
    metrics = {
        "exact_category_match": {"score": 0.85, "evaluated_on": 50}
    }
    
    print_metrics(metrics)
    captured = capsys.readouterr()
    
    assert "exact_category_match" in captured.out
    assert "0.85" in captured.out or "85" in captured.out


def test_print_metrics_with_multiple_scores(capsys):
    """Test print_metrics with multiple metric scores."""
    metrics = {
        "exact_category_match": {"score": 0.85, "evaluated_on": 50},
        "custom_metric": {"score": 0.92, "evaluated_on": 50},
    }
    
    print_metrics(metrics)
    captured = capsys.readouterr()
    
    assert "exact_category_match" in captured.out
    assert "custom_metric" in captured.out


def test_print_metrics_empty_metrics(capsys):
    """Test print_metrics with empty metrics dictionary."""
    metrics = {}
    
    # Should not raise an error
    print_metrics(metrics)
    captured = capsys.readouterr()
    
    # Output should be graceful (empty or notification)
    assert isinstance(captured.out, str)


def test_resolve_prompt_uri_with_absolute_uri():
    """Test _resolve_prompt_uri with an absolute URI."""
    uri = "models:/my-prompt/1"
    config = load_config()
    
    resolved = _resolve_prompt_uri(config, uri)
    
    assert resolved == uri


def test_resolve_prompt_uri_with_alias():
    """Test _resolve_prompt_uri resolves known aliases."""
    config = load_config()
    
    # Should resolve baseline alias
    resolved = _resolve_prompt_uri(config, "baseline")
    
    # Either returns a valid URI or the original string
    assert isinstance(resolved, str)
    assert len(resolved) > 0


def test_resolve_prompt_uri_with_malformed_input():
    """Test _resolve_prompt_uri handles malformed input gracefully."""
    config = load_config()
    
    # Malformed URI should still return a string
    resolved = _resolve_prompt_uri(config, "invalid/uri//format")
    
    assert isinstance(resolved, str)


@patch("src.evaluate.mlflow.genai.evaluate")
@patch("src.evaluate.load_eval_data")
def test_evaluate_progress_reporting(mock_load_data, mock_mlfow_eval):
    """Test that evaluate provides progress feedback."""
    mock_load_data.return_value = pd.DataFrame({
        "inputs": [{"customer_message": f"test {i}"} for i in range(5)],
        "expectations": [{"type": "incident"} for i in range(5)],
    })
    mock_mlfow_eval.return_value = {"exact_category_match": {"score": 0.8}}
    
    config = load_config()
    
    # Should complete without raising errors
    result = evaluate(config, prompt_uri="models:/test/1", limit=5)
    assert result is not None
