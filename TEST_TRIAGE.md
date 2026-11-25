# Triage of the 7 failing security tests

`automated_security_tests/` was added in commit 5eaac58 and the tests were generated with an AI assistant (IBM Bob; see `bob_sessions/`). On `main` the suite ran 117 passed, 7 failed. Each failure was investigated and classified. No assertion was weakened, no function was added to satisfy a test, and no failure is reported as a pass.

| Test (original) | Classification | Resolution |
| --- | --- | --- |
| `test_route_intent_returns_yes_on_api_exception` | Outdated or planned interface: `llm_client.route_intent` has never existed in the repo history. | Replaced by `test_llm_outage_fails_closed_when_extractive_fallback_disabled` and `test_llm_outage_uses_labelled_extractive_fallback_when_enabled`, which test what `generation.answer_question` really does when the LLM is down. Note the old test expected fail-open ("yes"); the real system fails closed (abstains) unless the extractive fallback is explicitly enabled. |
| `test_route_intent_returns_no_for_out_of_scope` | Same | Replaced by `test_out_of_scope_query_abstains_without_calling_the_llm`. |
| `test_rewrite_query_returns_original_on_api_failure` | Outdated or planned interface: `rewrite_query` has never existed. | Covered by the outage tests above. No query-rewriting requirement exists in the project documentation. |
| `test_rewrite_query_returns_original_when_no_api_key` | Same | Replaced by `test_missing_api_key_never_calls_the_llm`. |
| `test_result_count_capped_at_k` | Invalid test assumption: the mocked vector store ignored `k`, and the real store honours it. | Rewritten as `test_k_is_forwarded_to_the_vector_store` (asserts `k` is passed to the store and at most `k` results come back). |
| `test_parent_chunk_deduplication_keeps_only_first` | Pending feature: parent/child chunking (`parent_chunk_id`) is not produced by `ingest.py` nor read by `retrieve_with_citation`. | Kept unchanged as `xfail(strict=True)` with an explicit reason. |
| `test_parent_text_takes_precedence_over_page_content` | Pending feature: same, for `parent_text`. | Kept unchanged as `xfail(strict=True)` with an explicit reason. |

None of the seven was a defect in shipped security behaviour.

## Pending-feature follow-up

Implement parent/child chunking (ingest stores `parent_chunk_id` and `parent_text`; retrieval deduplicates by parent and prefers `parent_text`). Until then the two tests stay `xfail(strict=True)`: they are counted as expected failures, never as passes, and they fail the run if the feature appears without the mark being removed.

## Result

CI runs `pytest tests/` and `pytest automated_security_tests/`. Local run (Python 3.11, `requirements.lock`): 137 passed, 2 xfailed, 0 failed (139 tests collected across `tests/` and `automated_security_tests/`).
