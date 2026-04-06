"""Prompt helpers for the ticket classifier demo."""

from __future__ import annotations

from .config import Config

# Local fallback used when a registered MLflow prompt URI is not supplied.
BASE_PROMPT_TEMPLATE = """\
Classify the following customer support message into **exactly one** of these categories:

- Incident  : unexpected issue requiring immediate attention
- Request   : routine inquiry or service request
- Problem   : underlying / systemic issue causing multiple incidents
- Change    : planned change, update or configuration request

Customer message:
{{customer_message}}

Return **only** the category name.
Allowed answers: Incident, Request, Problem, Change
No explanation. No extra text.
"""


def load_prompt_uri(config: Config, version: str = "latest") -> str:
    """Build the MLflow URI for the registered prompt."""
    return f"prompts:/{config.prompt_template_name}@{version}"
