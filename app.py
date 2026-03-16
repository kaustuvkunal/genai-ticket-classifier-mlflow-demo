"""Entry point used when deploying to Hugging Face Spaces.

This file is intentionally minimal; it imports the Gradio app from the
packaged library so that the application can be launched directly.
"""

from genai_ticket_classifier.app import create_app


app = create_app()


if __name__ == "__main__":
    app.launch()
