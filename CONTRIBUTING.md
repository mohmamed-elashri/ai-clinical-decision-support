# Contributing

Thank you for helping improve AI Clinical Decision Support Lite.

## Before opening a pull request

- Explain the problem and the smallest complete change that addresses it.
- Do not commit `.env` files, API keys, credentials, local vector stores, or patient data.
- Add or update focused tests for behavior changes.
- Update documentation and measured benchmark claims when they change.
- Run `python -m unittest discover -s tests -v` locally. In CI, the equivalent command is `pytest tests/ -v`.
- Run `ruff check app.py config.py generation.py llm_client.py serve.py --select E9,F63,F7,F82` and `python -m compileall -q -x '(^|/)(scratch|notebooks)/' .` when Ruff is available.

## Clinical and safety changes

Changes that affect retrieval, abstention, provenance, prompts, schemas, or clinical wording must include:

1. The intended safety behavior and its boundary conditions.
2. A regression test or an evaluation-set update.
3. Any changed benchmark result and the command used to reproduce it.

This project is a research and engineering artifact, not a substitute for clinician review or a regulated clinical validation process.

## Pull requests

Use a clear title, describe validation performed, identify known limitations, and keep unrelated formatting or generated artifacts out of the change.
