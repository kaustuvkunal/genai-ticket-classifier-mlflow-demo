"""Lightweight web app for hosting the ticket classifier.

This module is intended for deployment as a Hugging Face Space (Gradio).
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import gradio as gr

from .config import Config, load_config
from .predict import predict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            logger.info(f"Classifying message: {message[:50]}...")
            if not message or not message.strip():
                result = "(enter a message to classify)"
                logger.info(f"Empty message, returning: {result}")
                return result
            try:
                result = predict(config, message)
                logger.info(f"Classification result: {result}")
                return result.strip()
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Classification error: {error_msg}")
                # Return a more user-friendly error message
                if "api_key" in error_msg.lower() or "groq" in error_msg.lower():
                    return f"❌ API Error: Check GROQ_API_KEY in .env"
                else:
                    return f"❌ Error: {error_msg[:100]}"

        txt.change(_classify, inputs=[txt], outputs=[out])
        gr.Button("Classify").click(_classify, inputs=[txt], outputs=[out])

        gr.Markdown(
            """---
            This demo uses Groq API for classification.
            """
        )

    # Configure auth for Hugging Face Spaces
    return demo


app = create_app()


if __name__ == "__main__":
    # Set up authentication
    auth = None
    
    # Option 1: Use environment variables HF_APP_USERNAME and HF_APP_PASSWORD
    hf_username = os.getenv("HF_APP_USERNAME")
    hf_password = os.getenv("HF_APP_PASSWORD")
    
    if hf_username and hf_password:
        auth = (hf_username, hf_password)
        print(f"✅ Using environment variable auth: {hf_username}")
    else:
        # Option 2: Use default credentials if env vars not set
        default_username = "demouser"
        default_password = os.getenv("PASSWD", "demopass")
        auth = (default_username, default_password)
        print(f"✅ Using default auth: {default_username} (password from PASSWD env var or 'demopass')")
    
    # Add queue support for better handling of concurrent requests
    app.queue()
    
    # Launch with authentication
    app.launch(auth=auth, css="body { font-family: system-ui; }")

