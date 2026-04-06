import importlib
import pytest

predict_module = importlib.import_module("src.predict")
from src.predict import predict_from_inputs, _build_messages, _get_prompt_template


def test_predict_from_inputs_calls_predict(monkeypatch):
    called = {"args": None, "kwargs": None}

    def fake_predict(config, customer_message: str, **kwargs) -> str:
        called["args"] = (config, customer_message)
        called["kwargs"] = kwargs
        return "incident"

    monkeypatch.setattr(predict_module, "predict", fake_predict)

    class DummyConfig:
        pass

    result = predict_from_inputs(DummyConfig(), {"customer_message": "hello"})

    assert result == "incident"
    assert called["args"][1] == "hello"
    assert called["kwargs"]["prompt_uri"] is None
    assert called["kwargs"]["prompt_template"] is None
    assert called["kwargs"]["client"] is None


def test_predict_from_inputs_missing_message():
    with pytest.raises(ValueError):
        predict_from_inputs(None, {})


def test_build_messages():
    """Test that messages are correctly formatted for LLM API."""
    prompt = "You are a classifier."
    message = "Customer issue here."

    messages = _build_messages(prompt, message)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == prompt
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == message


def test_get_prompt_template_from_direct_input():
    """Test that direct prompt_template parameter is used first."""
    custom_prompt = "Custom prompt for testing."
    prompt_uri = "prompts:/some/uri"

    result = _get_prompt_template(
        prompt_uri=prompt_uri,
        prompt_template=custom_prompt
    )

    assert result == custom_prompt


def test_get_prompt_template_from_fallback():
    """Test that default BASE_PROMPT_TEMPLATE is used as fallback."""
    from src.prompt import BASE_PROMPT_TEMPLATE

    result = _get_prompt_template()

    assert result == BASE_PROMPT_TEMPLATE
    assert "Incident" in result
    assert "Request" in result
    assert "Problem" in result
    assert "Change" in result


def test_predict_from_inputs_with_traced_false(monkeypatch):
    """Test that traced=False calls predict_once instead of predict."""
    called_once = {"count": 0}
    called_traced = {"count": 0}

    original_predict_once = predict_module._predict_once
    original_predict = predict_module.predict

    def fake_predict_once(*args, **kwargs):
        called_once["count"] += 1
        return "incident"

    def fake_predict(*args, **kwargs):
        called_traced["count"] += 1
        return "incident"

    monkeypatch.setattr(predict_module, "_predict_once", fake_predict_once)
    monkeypatch.setattr(predict_module, "predict", fake_predict)

    class DummyConfig:
        pass

    # With traced=False, should call _predict_once
    predict_from_inputs(
        DummyConfig(),
        {"customer_message": "test"},
        traced=False
    )
    assert called_once["count"] == 1
    assert called_traced["count"] == 0

    # Reset counters
    called_once["count"] = 0
    called_traced["count"] = 0

    # With traced=True, should call predict
    predict_from_inputs(
        DummyConfig(),
        {"customer_message": "test"},
        traced=True
    )
    assert called_traced["count"] == 1


def test_predict_batch_not_implemented():
    """Test that predict_batch function exists and works."""
    from src.predict import predict_batch
    # This is a simple check that the function is callable
    assert callable(predict_batch)
