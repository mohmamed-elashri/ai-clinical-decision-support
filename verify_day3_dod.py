"""Day 3 Definition of Done — grounded generation + abstain guard."""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config
from query import answer

FAILURES: list[str] = []
CHECKS: list[str] = []


def ok(msg: str) -> None:
    CHECKS.append(f"PASS  {msg}")


def fail(msg: str) -> None:
    FAILURES.append(f"FAIL  {msg}")


def main() -> int:
    root = Path(__file__).resolve().parent
    required = [
        root / "generation.py",
        root / "query.py",
        root / "retrieval.py",
        root / "verify_day3_dod.py",
    ]
    for path in required:
        if path.exists():
            ok(f"{path.name} present")
        else:
            fail(f"{path.name} missing")

    in_scope_question = "What is the target blood pressure for a patient with cardiovascular disease?"
    response = answer(in_scope_question, k=3)
    if response["status"] == "grounded":
        ok("in-scope question yields grounded answer")
    else:
        fail(f"in-scope query did not ground: {response['status']}")

    if response.get("citations"):
        ok(f"grounded answer contains {len(response['citations'])} citation(s)")
    else:
        fail("grounded answer missing citations")

    if response.get("top_score", 0.0) >= config.OUT_OF_SCOPE_MAX_SCORE:
        ok(f"in-scope top score is acceptable: {response['top_score']:.3f}")
    else:
        fail(f"in-scope top score unexpectedly below threshold: {response.get('top_score', 0.0):.3f}")

    # Validate schema conformity
    import jsonschema
    import json
    schema_path = root / "schema" / "response_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    validator = jsonschema.Draft7Validator(schema)
    errors = list(validator.iter_errors(response))
    if not errors:
        ok("in-scope response strictly conforms to response_schema.json")
    else:
        fail(f"response violates schema: {[e.message for e in errors]}")

    out_of_scope = "What is the recommended screening interval for breast cancer?"
    abstain = answer(out_of_scope, k=3)
    if abstain["status"] == "abstain":
        ok("out-of-scope query abstains correctly")
    else:
        fail(f"out-of-scope query should abstain, got {abstain['status']}")

    if abstain.get("top_score", 0.0) < config.OUT_OF_SCOPE_MAX_SCORE:
        ok(f"abstain guard triggered below threshold: {abstain['top_score']:.3f}")
    else:
        fail(f"abstain guard threshold not enforced: {abstain['top_score']:.3f}")

    paraphrase = "What blood pressure target should be used for someone with heart disease?"
    paraphrase_resp = answer(paraphrase, k=3)
    if paraphrase_resp["status"] == "grounded":
        ok("paraphrase remains grounded with the default dense retrieval path")
    else:
        fail(f"paraphrase query unexpectedly abstained: {paraphrase_resp['status']}")

    print("\n".join(CHECKS))
    if FAILURES:
        print("\n".join(FAILURES), file=sys.stderr)
        print(f"\n{len(FAILURES)} Day 3 check(s) FAILED.", file=sys.stderr)
        return 1
    print(f"\nAll {len(CHECKS)} Day 3 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
