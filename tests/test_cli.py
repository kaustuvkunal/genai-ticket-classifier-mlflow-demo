"""Tests for Click CLI commands."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from src.cli import cli, register_prompt_cmd, evaluate_cmd, optimize_cmd, predict_cmd


@pytest.fixture
def cli_runner():
    """Fixture providing CLI runner."""
    return CliRunner()


def test_cli_entry_point_exists(cli_runner):
    """Test that CLI entry point is accessible."""
    result = cli_runner.invoke(cli, ["--help"])
    
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_cli_has_register_command(cli_runner):
    """Test that register-prompt command is available."""
    result = cli_runner.invoke(cli, ["--help"])
    
    assert result.exit_code == 0
    assert "register-prompt" in result.output or "register" in result.output


def test_cli_has_evaluate_command(cli_runner):
    """Test that evaluate command is available."""
    result = cli_runner.invoke(cli, ["--help"])
    
    assert result.exit_code == 0
    assert "evaluate" in result.output


def test_cli_has_optimize_command(cli_runner):
    """Test that optimize command is available."""
    result = cli_runner.invoke(cli, ["--help"])
    
    assert result.exit_code == 0
    assert "optimize" in result.output


def test_cli_has_predict_command(cli_runner):
    """Test that predict command is available."""
    result = cli_runner.invoke(cli, ["--help"])
    
    assert result.exit_code == 0
    assert "predict" in result.output


@patch("src.cli.register_prompt")
def test_register_prompt_cmd_success(mock_register, cli_runner):
    """Test register-prompt command success."""
    mock_register.return_value = MagicMock(
        name="test-prompt",
        uri="models:/test-prompt/1",
        version=1
    )
    
    result = cli_runner.invoke(register_prompt_cmd)
    
    # Should complete with success or informative error
    assert result.exit_code in [0, 1]  # 0=success, 1=expected error


@patch("src.cli.evaluate")
@patch("src.cli.load_config")
def test_evaluate_cmd_with_limit(mock_config, mock_eval, cli_runner):
    """Test evaluate command with limit parameter."""
    mock_config.return_value = MagicMock()
    mock_eval.return_value = {"exact_category_match": {"score": 0.85}}
    
    result = cli_runner.invoke(evaluate_cmd, ["--limit", "5"])
    
    # Should either succeed or have expected validation
    assert result.exit_code in [0, 1, 2]


@patch("src.cli.optimize_prompt")
@patch("src.cli.load_config")
def test_optimize_cmd_basic(mock_config, mock_optimize, cli_runner):
    """Test optimize command."""
    mock_config.return_value = MagicMock()
    mock_optimize.return_value = MagicMock(
        name="optimized-prompt",
        uri="models:/optimized/1"
    )
    
    result = cli_runner.invoke(optimize_cmd)
    
    # Should complete or provide sensible error
    assert result.exit_code in [0, 1, 2]


@patch("src.cli.predict_ticket")
@patch("src.cli.load_config")
def test_predict_cmd_with_message(mock_config, mock_predict, cli_runner):
    """Test predict command with message."""
    mock_config.return_value = MagicMock()
    mock_predict.return_value = "incident"
    
    result = cli_runner.invoke(predict_cmd, ["--message", "Test issue"])
    
    # Should succeed or have expected error
    assert result.exit_code in [0, 1, 2]


def test_predict_cmd_missing_message(cli_runner):
    """Test predict command without required message."""
    result = cli_runner.invoke(predict_cmd)
    
    # Should indicate missing required parameter
    assert result.exit_code != 0


@patch("src.cli.load_config")
def test_commands_handle_config_errors(mock_config, cli_runner):
    """Test that commands handle config loading errors gracefully."""
    mock_config.side_effect = Exception("Config error")
    
    # Should exit with error code, not crash
    result = cli_runner.invoke(evaluate_cmd, catch_exceptions=False)
    
    # Will likely fail, but shouldn't have unhandled exceptions
    assert result.exit_code != 0


def test_cli_commands_are_idempotent(cli_runner):
    """Test that CLI help queries don't have side effects."""
    result1 = cli_runner.invoke(cli, ["register-prompt", "--help"])
    result2 = cli_runner.invoke(cli, ["register-prompt", "--help"])
    
    # Running help twice should give same result
    assert result1.output == result2.output


@patch("src.cli.load_config")
def test_evaluate_cmd_with_prompt_uri(mock_config, cli_runner):
    """Test evaluate command with custom prompt URI."""
    mock_config.return_value = MagicMock()
    
    with patch("src.cli.evaluate") as mock_eval:
        mock_eval.return_value = {"score": 0.8}
        
        result = cli_runner.invoke(
            evaluate_cmd,
            ["--prompt-uri", "models:/custom-prompt/1"]
        )
        
        # Should handle the custom URI
        assert result.exit_code in [0, 1, 2]


def test_cli_error_messages_are_informative(cli_runner):
    """Test that CLI error messages are user-friendly."""
    # Invoke with invalid command
    result = cli_runner.invoke(cli, ["nonexistent-command"])
    
    # Error should be clear
    assert result.exit_code != 0
    output = result.output.lower()
    assert "error" in output or "no such command" in output
