# Running the AI Clinical Decision Support Web App

## Prerequisites

1. **Python 3.11+** installed and on your `PATH`.
2. The Chroma vector index is already built (`vectorstore/` directory exists).
   If not, run: `python ingest.py`
3. A valid NVIDIA NIM API key in your `.env` file:

```dotenv
NV_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
NV_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1/chat/completions
NV_MODEL=meta/llama-3.1-8b-instruct
```

4. Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Mode 1 — Two Separate Processes (recommended for development)

### Terminal 1 — Start the Flask backend

```bash
python app.py
```

The backend starts on **http://localhost:5000**.

Optional environment overrides:

| Variable       | Default               | Description                         |
|----------------|-----------------------|-------------------------------------|
| `APP_PORT`     | `5000`                | Flask listen port                   |
| `APP_DEBUG`    | `0`                   | Set `1` for debug mode (dev only)   |
| `APP_ENV`      | `development`         | Set `production` in deployed environments |
| `CORS_ORIGINS` | local Gradio origins  | Comma-separated allowed origins     |

### Terminal 2 — Start the Gradio frontend

```bash
python frontend.py
```

The UI opens at **http://localhost:7860**.

## Production API

Do not use `python app.py` as the production server. Install the dependencies
and serve the WSGI application with Gunicorn behind a TLS reverse proxy:

```bash
pip install -r requirements.txt
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 wsgi:app
```

Containerized deployment with Redis:

```bash
docker compose up --build -d
docker compose logs -f api
```

The compose setup mounts the existing `vectorstore/` read-only and persists
Redis data in a named volume. Keep `.env` outside images and inject it through
the deployment secret store.

For reproducible installs, use the pinned direct-dependency baseline:

```bash
pip install -r requirements.lock
```

Production requirements:

- Set `APP_ENV=production`, `API_AUTH_TOKEN` to a long random secret, and
  explicit `CORS_ORIGINS`.
- Keep `APP_DEBUG=0` and `ALLOW_EXTRACTIVE_FALLBACK=0`.
- Provide the persisted vector index through a managed volume or build it
  deterministically before starting the API.
- Terminate TLS and enforce request size/time limits at the reverse proxy.
- Set `REDIS_URL` to a private Redis instance; it is required in production so
  rate limits are shared across Gunicorn workers.

Use `/api/live` for liveness probes and `/api/health` for readiness probes;
the latter initializes and validates the vector index.

Optional environment overrides:

| Variable        | Default                   | Description               |
|-----------------|---------------------------|---------------------------|
| `FLASK_API_URL` | `http://localhost:5000`   | URL of the Flask backend  |
| `FLASK_API_KEY` | empty                     | Must match `API_AUTH_TOKEN` when auth is enabled |
| `GRADIO_PORT`   | `7860`                    | Gradio listen port        |
| `GRADIO_SHARE`  | `0`                       | Set `1` only for temporary public demos |

---

## Mode 2 — Single Process (optional, for demos)

You can serve both the Flask API and the Gradio UI from a single Python
process using `gradio.mount_gradio_app`:

```bash
python -c "import gradio as gr; from app import app; from frontend import demo; gr.mount_gradio_app(app, demo, path='/ui'); app.run(host='127.0.0.1', port=5000)"
```

> **Note:** The command mounts the Gradio `demo` object onto the Flask app
> at the `/ui` route. The API endpoints
> remain at `/api/*` and the UI is accessible at `http://localhost:5000/ui`.

For production, set `API_AUTH_TOKEN` and use a production WSGI server.
When the frontend runs as a separate process, set `FLASK_API_KEY` to the same
secret so its backend requests are authenticated.

---

## Manual Test Checklist

### 1. Backend health check

```bash
curl http://localhost:5000/api/health
```

Expected:
```json
{"status": "ok", "chunks_indexed": 190}
```

### 2. In-scope question → grounded answer with citations

```bash
curl -s -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the target blood pressure for a patient with cardiovascular disease?"}' \
  | python -m json.tool
```

Expected:
```json
{
  "status": "grounded",
  "confidence": "confident",
  "citations": [{"document_name": "...", "page_number": 28}],
  "generation_method": "llm"
}
```

### 3. Out-of-scope question → abstain response

```bash
curl -s -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the recommended screening interval for breast cancer?"}' \
  | python -m json.tool
```

Expected:
```json
{
  "status": "abstain",
  "confidence": "insufficient",
  "citations": [],
  "generation_method": "none"
}
```

The Gradio UI displays: **"Outside guideline scope"** — no fabricated content.

### 4. Empty question → HTTP 400

```bash
curl -s -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": ""}' \
  | python -m json.tool
```

Expected:
```json
{"error": "Question must not be empty."}
```

### 5. Existing verification scripts (must all still pass)

```bash
python verify_dod.py          # Day 1: 12/12
python verify_day2_dod.py     # Day 2: 21/21
python verify_day3_dod.py     # Day 3: 11/11
```

---

## Security Notes

- **Never** commit your `.env` file — it is already in `.gitignore`.
- `app.py` will exit immediately at startup if `NV_API_KEY` is not set.
- The `NV_API_KEY` is never logged, echoed, or returned in any API response.
- Questions are hard-capped at 1000 characters to reduce prompt-injection risk.
