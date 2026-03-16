# genai-ticket-classifier-mlflow-demo

GenAI ticket classifier demo using **MLflow GenAI** 


## Features

- Registers a reusable MLflow prompt for ticket classification
- Runs evaluation against a labeled dataset using `mlflow.genai.evaluate`
- Optimizes prompts using `mlflow.genai.optimize_prompts`
- Provides a simple CLI for registration, evaluation, optimization, and prediction

## Getting started

### 1) Set up a virtual environment (recommended)

Create and activate a virtual environment to isolate dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3) Configure credentials

Copy the `.env.example` file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` and set `GROQ_API_KEY` (required). Optionally adjust MLflow settings.

### 4) Start an MLflow tracking server (local)

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

### 5) Run the demo CLI

Register the prompt:

```bash
ticket-classifier register-prompt
```

Run a baseline evaluation:

```bash
ticket-classifier evaluate
```

Optimize the prompt:

```bash
ticket-classifier optimize
```

Predict a single message:

```bash
ticket-classifier predict "My service is down and I need help"
```

## 6) Run unit tests

```bash
pytest
```

## Deploy to Hugging Face Spaces (with login)

This project includes a Gradio app (`app.py`) designed for deployment to a Hugging Face Space.

### 5a) Prepare the Space

1. Create a new Space on Hugging Face and choose **Gradio** as the runtime.
2. Push this repo to the Space (or connect it via Git).

### 5b) Configure secrets (login)

In the Space settings, add the following secrets (under **Secrets**):

- `GROQ_API_KEY` — your Groq API key
- `MLFLOW_TRACKING_URI` — e.g., `http://127.0.0.1:5000` (if using a remote server)
- `HF_APP_USERNAME` — the username required to access the app
- `HF_APP_PASSWORD` — the password required to access the app

> If `HF_APP_USERNAME` and/or `HF_APP_PASSWORD` are unset, the app will run without login.

### 5c) Launch

The Space automatically runs `python app.py` on startup. Once the build completes, visit the Space URL and log in using the username/password configured above.

### 5d) Local development

You can also run the app locally:

```bash
python app.py
```

If you want to require auth locally as well, set the same environment variables:

```bash
export HF_APP_USERNAME=myuser
export HF_APP_PASSWORD=mypassword
python app.py
```

## Project layout

- `src/genai_ticket_classifier/` – core library code
- `notebooks/` – exploratory notebook (legacy)
- `.env.example` – environment variable template
- `requirements.txt` – pinned runtime dependencies
- `pyproject.toml` – project metadata and CLI entry point
