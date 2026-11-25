# Attribution

AI Clinical Decision Support Lite is a team project by **Insight AI**.

- Upstream repository: https://github.com/MohamedAwadMoneer/ai-clinical-decision-support
- License: MIT, copyright (c) 2026 Mohamed Awad Moneer. The original `LICENSE` is kept unchanged. Copyright and credit for the upstream code remain with the upstream author and the team.
- This repository is a copy maintained by Mohamed Elashri (`mohmamed-elashri`), a team member. It is not a GitHub fork.

## Mohamed Elashri's contributions

The list below is Mohamed's own description of his work. The "Evidence" column notes what can be checked in this repository. The upstream history was committed under a shared team identity, so most individual authorship cannot be confirmed from commits.

| Area | Evidence in this repository |
|------|-----------------------------|
| Flask REST API with API-key authentication | `app.py` (constant-time key comparison, `X-API-Key`). Individual authorship not shown by history |
| Redis-based rate limiting | `app.py`, `docker-compose.yml`. Individual authorship not shown by history |
| Hybrid retrieval (BM25 + dense, RRF) | `hybrid_retriever.py`, `benchmark_30.py`, `BENCHMARK_30_REPORT.md` |
| RAG, citations, JSON Schema validation | `generation.py`, `llm_client.py`, `schema/response_schema.json` |
| Retrieval evaluation (paraphrase PageRecall@3 from 80% to 100% on a 30-question set) | Reported in `README.md` and `BENCHMARK_30_REPORT.md` |
| API contract tests | `tests/test_contracts.py` |
| Docker, Docker Compose, GitHub Actions | `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` |
| Automated security tests | `automated_security_tests/` (commit `5eaac58`, authored by Mohamed Elashri; generated with an AI coding assistant and reviewed by him) |

Team roles other than the above are not documented here. Corrections and additions from other team members are welcome through pull requests.
