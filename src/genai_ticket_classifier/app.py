"""Lightweight web app for hosting the ticket classifier.

This module is intended for deployment as a Hugging Face Space (Gradio).
"""

from __future__ import annotations

import os
from typing import Optional

import gradio as gr

from .config import Config, load_config
from .predict import predict


def _build_auth() -> Optional[tuple[str, str]]:
    """Build a Gradio basic auth tuple from environment variables.

    When deploying to Hugging Face Spaces, set these secrets in the UI:
      - HF_APP_USERNAME
      - HF_APP_PASSWORD

    Leaving either value blank disables auth.
    """

    username = os.getenv("HF_APP_USERNAME")
    password = os.getenv("HF_APP_PASSWORD")
    if not username or not password:
        return None
    return (username, password)


def create_app(config: Optional[Config] = None) -> gr.Blocks:
    """Create a Gradio interface for the ticket classifier."""

    config = config or load_config()
    auth = _build_auth()

    with gr.Blocks(title="GenAI Ticket Classifier") as demo:
        gr.Markdown("""
        # Ticket classifier

        Enter a customer message and the model will classify it into one of:
        `Incident`, `Request`, `Problem`, or `Change`.
        """)

        with gr.Row():
            txt = gr.TextArea(label="Customer message", placeholder="Describe the customer issue...")
            out = gr.Textbox(label="Predicted category")

        def _classify(message: str) -> str:
            if not message or not message.strip():
                return "(enter a message to classify)"
            return predict(config, message)

        txt.change(_classify, inputs=[txt], outputs=[out])
        gr.Button("Classify").click(_classify, inputs=[txt], outputs=[out])

        gr.Markdown(
            """---
            This demo uses MLflow GenAI and a registered prompt.
            """
        )

    # Configure auth for Hugging Face Spaces
    return demo


app = create_app()


if __name__ == "__main__":
    auth = _build_auth()
    app.launch(auth=auth, css="body { font-family: system-ui; }")
