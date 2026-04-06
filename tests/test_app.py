"""Tests for Gradio app endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from app import predict_with_groq, predict_with_openai


@patch("app.groq_client")
@patch("app.load_config")
def test_predict_with_groq_returns_string(mock_config, mock_groq):
    """Test that predict_with_groq returns a string prediction."""
    # Mock the Groq client response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="incident"))]
    mock_groq.chat.completions.create.return_value = mock_response
    
    # Mock config
    mock_cfg = MagicMock()
    mock_cfg.groq_api_key = "test-key"
    mock_config.return_value = mock_cfg
    
    result = predict_with_groq("What is the issue?")
    
    assert isinstance(result, str)
    assert len(result) > 0


@patch("app.openai_client")
@patch("app.load_config")
def test_predict_with_openai_returns_string(mock_config, mock_openai):
    """Test that predict_with_openai returns a string prediction."""
    # Mock the OpenAI Responses API response
    mock_response = MagicMock()
    mock_response.output_text = "request"
    mock_openai.responses.create.return_value = mock_response
    
    # Mock config
    mock_cfg = MagicMock()
    mock_cfg.openai_api_key = "test-key"
    mock_config.return_value = mock_cfg
    
    result = predict_with_openai("What is the feature request?")
    
    assert isinstance(result, str)
    assert len(result) > 0


@patch("app.groq_client")
@patch("app.load_config")
def test_predict_with_groq_error_handling(mock_config, mock_groq):
    """Test error handling in predict_with_groq."""
    # Mock a connection error
    mock_groq.chat.completions.create.side_effect = Exception("API error")
    
    mock_cfg = MagicMock()
    mock_config.return_value = mock_cfg
    
    # Should raise or return error message
    with pytest.raises(Exception):
        predict_with_groq("Test message")


@patch("app.openai_client")
@patch("app.load_config")
def test_predict_with_openai_error_handling(mock_config, mock_openai):
    """Test error handling in predict_with_openai."""
    # Mock a connection error
    mock_openai.responses.create.side_effect = Exception("API error")
    
    mock_cfg = MagicMock()
    mock_config.return_value = mock_cfg
    
    # Should raise or return error message
    with pytest.raises(Exception):
        predict_with_openai("Test message")


@patch("app.groq_client")
@patch("app.load_config")
def test_predict_with_groq_empty_input(mock_config, mock_groq):
    """Test predict_with_groq handles empty input."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=""))]
    mock_groq.chat.completions.create.return_value = mock_response
    
    mock_cfg = MagicMock()
    mock_config.return_value = mock_cfg
    
    result = predict_with_groq("")
    
    # Should return empty string or None gracefully
    assert isinstance(result, (str, type(None)))


@patch("app.openai_client")
@patch("app.load_config")
def test_predict_with_openai_empty_input(mock_config, mock_openai):
    """Test predict_with_openai handles empty input."""
    mock_response = MagicMock()
    mock_response.output_text = ""
    mock_openai.responses.create.return_value = mock_response
    
    mock_cfg = MagicMock()
    mock_config.return_value = mock_cfg
    
    result = predict_with_openai("")
    
    # Should return empty string or None gracefully
    assert isinstance(result, (str, type(None)))


@patch("app.groq_client")
@patch("app.load_config")
def test_predict_with_groq_long_input(mock_config, mock_groq):
    """Test predict_with_groq handles long customer message."""
    long_message = "X" * 5000  # 5000 character message
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="problem"))]
    mock_groq.chat.completions.create.return_value = mock_response
    
    mock_cfg = MagicMock()
    mock_config.return_value = mock_cfg
    
    result = predict_with_groq(long_message)
    
    assert isinstance(result, str)


@patch("app.openai_client")
@patch("app.load_config")
def test_predict_with_openai_uses_responses_api(mock_config, mock_openai):
    """Test that predict_with_openai uses Responses API (not chat.completions)."""
    mock_response = MagicMock()
    mock_response.output_text = "change"
    mock_openai.responses.create.return_value = mock_response
    
    mock_cfg = MagicMock()
    mock_config.return_value = mock_cfg
    
    predict_with_openai("Is this a change request?")
    
    # Verify responses.create was called (not chat.completions.create)
    mock_openai.responses.create.assert_called()
    mock_openai.chat.completions.create.assert_not_called()


def test_predict_with_groq_with_special_characters():
    """Test predict_with_groq handles special characters in input."""
    special_input = "Message with special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`"
    
    with patch("app.groq_client") as mock_groq:
        with patch("app.load_config") as mock_config:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="incident"))]
            mock_groq.chat.completions.create.return_value = mock_response
            
            mock_cfg = MagicMock()
            mock_config.return_value = mock_cfg
            
            # Should not raise an error
            result = predict_with_groq(special_input)
            assert isinstance(result, str)
