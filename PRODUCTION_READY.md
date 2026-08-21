# Production Readiness Checklist

The application has production-oriented API hardening, but it remains a
clinical decision-support aid and must not be used for unsupervised direct
patient care.

## Implemented

- WSGI entrypoint and Gunicorn deployment.
- Pinned direct dependencies in `requirements.lock`.
- Redis-backed shared rate limiting, required in production.
- API key authentication for protected endpoints.
- Explicit production CORS, request-size limits, and fail-fast configuration.
- Liveness (`/api/live`) and readiness (`/api/health`) probes.
- Request IDs, security headers, and audit metadata without question text or
  response content.
- Strict response schema and evidence provenance checks.
- Dockerfile and Docker Compose deployment with Redis healthchecks.
- Contract tests covering auth, probes, request limits, provenance, and retrieval
  score semantics.

## Required Before Deployment

- Build a clean Python 3.11 environment from `requirements.lock`.
- Store `NV_API_KEY`, `API_AUTH_TOKEN`, and Redis credentials in a secret manager.
- Put the API behind TLS and a reverse proxy with timeouts and network policy.
- Use managed/private Redis with authentication and backups.
- Configure centralized log retention and alerting for 5xx, 401/429, latency,
  readiness failures, and LLM failures.
- Complete clinician-reviewed evaluation for prompt injection, near-domain
  questions, abstention behavior, and citation correctness.
- Create a provenance manifest for indexed documents with checksums, source URLs,
  dates, and licensing information.
- Exercise disaster recovery for the vector index and Redis data.

## Verification

```bash
python -m unittest discover -s tests -v
python -m pip check
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 wsgi:app
```

For local containers, install Docker Desktop and run:

```bash
docker compose up --build -d
docker compose ps
```
