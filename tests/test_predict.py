import importlib
import pytest

predict_module = importlib.import_module("genai_ticket_classifier.predict")
from genai_ticket_classifier.predict import predict_from_inputs


def test_predict_from_inputs_calls_predict(monkeypatch):
    called = {"args": None}

    def fake_predict(config, customer_message: str) -> str:
        called["args"] = (config, customer_message)
        return "incident"

    monkeypatch.setattr(predict_module, "predict", fake_predict)

    class DummyConfig:
        pass

    result = predict_from_inputs(DummyConfig(), {"customer_message": "hello"})

    assert result == "incident"
    assert called["args"][1] == "hello"


def test_predict_from_inputs_missing_message():
    with pytest.raises(ValueError):
        predict_from_inputs(None, {})
