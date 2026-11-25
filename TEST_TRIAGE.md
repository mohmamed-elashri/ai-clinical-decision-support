# Triage of the 7 failing security tests

`automated_security_tests/` was added in commit 5eaac58 and the tests were generated with an AI assistant (IBM Bob; see `bob_sessions/`). On `main` the suite ran 117 passed, 7 failed. Nothing was deleted and no assertion was changed. Each failing test is now marked `xfail(strict=True)` with the reason below. If someone implements the missing behaviour, the strict marker turns the test into a failure until the mark is removed, so it cannot be forgotten.

| Test | Classification | Evidence |
| --- | --- | --- |
| `test_route_intent_returns_yes_on_api_exception` | Outdated or planned interface | `llm_client.route_intent` does not exist and never existed in the repo history. |
| `test_route_intent_returns_no_for_out_of_scope` | Outdated or planned interface | Same as above. |
| `test_rewrite_query_returns_original_on_api_failure` | Outdated or planned interface | `llm_client.rewrite_query` does not exist and never existed. |
| `test_rewrite_query_returns_original_when_no_api_key` | Outdated or planned interface | Same as above. |
| `test_parent_chunk_deduplication_keeps_only_first` | Test assumes an unimplemented feature | Parent/child chunking (`parent_chunk_id`) is not produced by `ingest.py` nor read by `retrieval.retrieve_with_citation`. |
| `test_parent_text_takes_precedence_over_page_content` | Test assumes an unimplemented feature | Same: `parent_text` is not used anywhere. |
| `test_result_count_capped_at_k` | Invalid test assumption | The mocked vector store ignores `k`; the real store honours it, so retrieval does not slice the list again. |

None of the 7 is a security defect in shipped code. No functions were added to satisfy the tests.

Open decision for the maintainers: implement the missing behaviour (then drop the xfail marks), or remove the tests that describe features the project does not have.

CI now runs `pytest tests/` and `pytest automated_security_tests/`. Local result: 132 passed, 7 xfailed.
