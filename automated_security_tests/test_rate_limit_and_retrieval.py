"""
test_rate_limit_and_retrieval.py
=================================
Comprehensive pytest suite covering:

  1. Rate-limiting logic in app.py (_rate_limit in-memory path, Redis path,
     and the protect_api before_request hook).
  2. ChromaDB retrieval helpers — retrieve_with_citation (retrieval.py),
     rrf_merge (hybrid_retriever.py), and the dense/BM25 retrieval paths.
  3. Flask API edge-case validation (/api/ask, /api/live, /api/health).

All external dependencies (ChromaDB, Redis, NVIDIA NIM, FlashRank, BM25) are
fully mocked — no real network calls or file I/O take place.

Run with:
    cd ai-clinical-decision-support-main
    python -m pytest automated_security_tests/test_rate_limit_and_retrieval.py -v
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import time
import types
from collections import defaultdict, deque
from threading import Lock
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_doc(
    page_content: str = "clinical text",
    metadata: dict | None = None,
):
    """Return a lightweight mock of a LangChain Document object."""
    doc = MagicMock()
    doc.page_content = page_content
    doc.metadata = metadata or {}
    return doc


def _make_vectordb(results: list[tuple] | None = None):
    """
    Return a mock vectordb whose similarity_search_with_relevance_scores method
    returns *results* — a list of (Document, float) tuples.

    Passes None for _collection so _collection_count() handles the missing attr.
    """
    vdb = MagicMock()
    vdb._collection = None
    vdb.similarity_search_with_relevance_scores.return_value = results or []
    return vdb


@pytest.fixture(autouse=True)
def _isolate_app_rate_state():
    """
    Reset all in-memory rate-limit state before every test so tests are
    independent even when they run in the same process.
    """
    import app as app_module

    original_times = app_module._request_times
    original_redis = app_module._redis_client
    original_redis_url = app_module._REDIS_URL

    app_module._request_times = defaultdict(deque)
    app_module._redis_client = None
    app_module._REDIS_URL = ""

    yield

    app_module._request_times = original_times
    app_module._redis_client = original_redis
    app_module._REDIS_URL = original_redis_url


@pytest.fixture()
def flask_client():
    """Provide a Flask test client with auth and rate-limit bypassed by default."""
    import app as app_module

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        yield client


# ===========================================================================
# Section 1 — Rate Limiting: in-memory path
# ===========================================================================


class TestRateLimitInMemory:
    """Tests for the in-memory (non-Redis) sliding-window rate limiter."""

    def _call(self, client_key: str = "1.2.3.4"):
        import app as app_module
        return app_module._rate_limit(client_key)

    # Test 1
    def test_first_request_is_allowed(self):
        """First request for a fresh client key must be allowed."""
        assert self._call("10.0.0.1") == "allowed"

    # Test 2
    def test_exactly_at_limit_is_allowed(self):
        """The N-th request where N == limit must still be allowed (inclusive boundary)."""
        import app as app_module

        limit = 5
        with patch.object(app_module, "_RATE_LIMIT_PER_MINUTE", limit):
            app_module._request_times = defaultdict(deque)
            results = [self._call("edge-client") for _ in range(limit)]
        assert all(r == "allowed" for r in results), results

    # Test 3
    def test_one_over_limit_is_rejected(self):
        """The (N+1)-th request within the window must be rejected as 'limited'."""
        import app as app_module

        limit = 5
        with patch.object(app_module, "_RATE_LIMIT_PER_MINUTE", limit):
            app_module._request_times = defaultdict(deque)
            for _ in range(limit):
                self._call("over-client")
            result = self._call("over-client")
        assert result == "limited"

    # Test 4
    def test_counter_resets_after_60_second_window(self):
        """After the 60-second window expires the counter resets and the next request is allowed."""
        import app as app_module

        limit = 2
        fake_now = 1_000.0

        with patch.object(app_module, "_RATE_LIMIT_PER_MINUTE", limit):
            app_module._request_times = defaultdict(deque)
            # Fill the window right to the limit.
            with patch("app.time.monotonic", return_value=fake_now):
                for _ in range(limit):
                    self._call("reset-client")
            # Advance time by exactly 60 seconds so all timestamps expire.
            with patch("app.time.monotonic", return_value=fake_now + 60.0):
                result = self._call("reset-client")
        assert result == "allowed"

    # Test 5
    def test_independent_counters_per_ip(self):
        """Two distinct client IPs must have completely independent counters."""
        import app as app_module

        limit = 2
        with patch.object(app_module, "_RATE_LIMIT_PER_MINUTE", limit):
            app_module._request_times = defaultdict(deque)
            # Exhaust client A.
            for _ in range(limit):
                self._call("client-A")
            assert self._call("client-A") == "limited"
            # Client B should still be at zero — must be allowed.
            assert self._call("client-B") == "allowed"

    # Test 6
    def test_none_remote_addr_falls_back_to_unknown_key(self):
        """If client_key is falsy the caller passes 'unknown'; must be handled gracefully."""
        import app as app_module

        # Simulate what protect_api does: request.remote_addr or "unknown"
        result = app_module._rate_limit("unknown")
        assert result == "allowed"

    # Test 7
    def test_rate_limit_of_1_second_request_is_limited(self):
        """With a limit of 1, the very next request must be rejected immediately."""
        import app as app_module

        with patch.object(app_module, "_RATE_LIMIT_PER_MINUTE", 1):
            app_module._request_times = defaultdict(deque)
            first = self._call("tiny-limit")
            second = self._call("tiny-limit")
        assert first == "allowed"
        assert second == "limited"

    # Test 8
    def test_zero_rate_limit_first_request_is_limited(self):
        """A rate limit of 0 means no requests are allowed at all."""
        import app as app_module

        with patch.object(app_module, "_RATE_LIMIT_PER_MINUTE", 0):
            app_module._request_times = defaultdict(deque)
            result = self._call("zero-limit")
        # len(timestamps) >= 0 is always True, so the first request is immediately limited.
        assert result == "limited"

    # Test 9
    def test_flask_returns_429_when_rate_limited(self):
        """protect_api must short-circuit with HTTP 429 and a JSON error body."""
        import app as app_module

        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as client:
            with patch.object(app_module, "_rate_limit", return_value="limited"):
                resp = client.get("/api/live")
        assert resp.status_code == 429
        data = resp.get_json()
        assert data is not None
        assert "error" in data
        assert "rate limit" in data["error"].lower()

    # Test 10
    def test_flask_returns_503_when_backend_unavailable(self):
        """protect_api must return HTTP 503 when _rate_limit returns 'unavailable'."""
        import app as app_module

        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as client:
            with patch.object(app_module, "_rate_limit", return_value="unavailable"):
                resp = client.get("/api/live")
        assert resp.status_code == 503
        data = resp.get_json()
        assert "unavailable" in data.get("error", "").lower()


# ===========================================================================
# Section 2 — Rate Limiting: Redis path
# ===========================================================================


class TestRateLimitRedis:
    """Tests for the Redis-backed rate limiter branch."""

    def _call_with_redis(
        self,
        redis_mock: MagicMock,
        client_key: str = "10.0.0.1",
        redis_url: str = "redis://localhost:6379",
    ) -> str:
        import app as app_module

        # Point _REDIS_URL at a fake URL so the Redis branch is taken.
        app_module._REDIS_URL = redis_url
        # Inject our mock as the pre-existing client so no real connection is made.
        app_module._redis_client = redis_mock
        return app_module._rate_limit(client_key)

    # Test 11
    def test_redis_first_call_returns_allowed(self):
        """When Redis returns count=1 the result must be 'allowed'."""
        redis_mock = MagicMock()
        redis_mock.incr.return_value = 1
        import app as app_module

        with patch.object(app_module, "_RATE_LIMIT_PER_MINUTE", 60):
            result = self._call_with_redis(redis_mock)
        assert result == "allowed"
        redis_mock.expire.assert_called_once()

    # Test 12
    def test_redis_over_limit_returns_limited(self):
        """When Redis count exceeds the per-minute limit the result must be 'limited'."""
        redis_mock = MagicMock()
        import app as app_module

        limit = 60
        redis_mock.incr.return_value = limit + 1
        with patch.object(app_module, "_RATE_LIMIT_PER_MINUTE", limit):
            result = self._call_with_redis(redis_mock)
        assert result == "limited"

    # Test 13
    def test_redis_exception_returns_unavailable(self):
        """Any exception from the Redis client must yield 'unavailable' (fail-open)."""
        redis_mock = MagicMock()
        redis_mock.incr.side_effect = Exception("connection refused")
        result = self._call_with_redis(redis_mock)
        assert result == "unavailable"

    # Test 14
    def test_redis_key_uses_sha256_digest_of_client_key(self):
        """The Redis bucket key must embed a SHA-256 hex digest of the client IP."""
        redis_mock = MagicMock()
        redis_mock.incr.return_value = 1

        client_key = "192.168.1.99"
        expected_digest = hashlib.sha256(client_key.encode("utf-8")).hexdigest()[:24]

        self._call_with_redis(redis_mock, client_key=client_key)

        # Verify that incr was called with a key containing the expected digest.
        called_key: str = redis_mock.incr.call_args[0][0]
        assert expected_digest in called_key, (
            f"Expected digest '{expected_digest}' not found in Redis key '{called_key}'"
        )


# ===========================================================================
# Section 3 — retrieve_with_citation (retrieval.py)
# ===========================================================================


class TestRetrieveWithCitation:
    """Unit tests for retrieval.retrieve_with_citation (dense path)."""

    def _retrieve(self, vectordb, question="What is hypertension?", k=3, **kwargs):
        from retrieval import retrieve_with_citation
        return retrieve_with_citation(vectordb, question, k=k, retrieval_mode="dense", **kwargs)

    # Test 15
    def test_k_zero_raises_value_error(self):
        """k=0 must raise ValueError — zero results is nonsensical."""
        from retrieval import retrieve_with_citation
        with pytest.raises(ValueError, match="positive integer"):
            retrieve_with_citation(_make_vectordb(), "q", k=0, retrieval_mode="dense")

    # Test 16
    def test_k_bool_raises_value_error(self):
        """k=False (bool) must raise ValueError — bool is an int subclass and must be rejected."""
        from retrieval import retrieve_with_citation
        with pytest.raises(ValueError, match="positive integer"):
            retrieve_with_citation(_make_vectordb(), "q", k=False, retrieval_mode="dense")

    # Test 17
    def test_k_string_raises_value_error(self):
        """k='3' (string) must raise ValueError — only native int is accepted."""
        from retrieval import retrieve_with_citation
        with pytest.raises(ValueError, match="positive integer"):
            retrieve_with_citation(_make_vectordb(), "q", k="3", retrieval_mode="dense")

    # Test 18
    def test_k_float_raises_value_error(self):
        """k=1.5 (float) must raise ValueError — floats are not valid k values."""
        from retrieval import retrieve_with_citation
        with pytest.raises(ValueError, match="positive integer"):
            retrieve_with_citation(_make_vectordb(), "q", k=1.5, retrieval_mode="dense")

    # Test 19
    def test_invalid_retrieval_mode_raises_value_error(self):
        """retrieval_mode='hybrid' must raise ValueError — only 'dense' and 'rrf' are valid."""
        from retrieval import retrieve_with_citation
        with pytest.raises(ValueError, match="retrieval_mode"):
            retrieve_with_citation(_make_vectordb(), "q", k=3, retrieval_mode="hybrid")

    # Test 20
    def test_retrieval_mode_whitespace_is_normalised(self):
        """' dense ' with surrounding whitespace must be treated as 'dense' (no error)."""
        vdb = _make_vectordb(results=[])
        from retrieval import retrieve_with_citation
        # Should not raise; whitespace is stripped via .strip().lower()
        result = retrieve_with_citation(vdb, "q", k=3, retrieval_mode=" dense ")
        assert isinstance(result, list)

    # Test 21
    def test_empty_vectordb_returns_empty_list(self):
        """If the vector store returns no results the output must be an empty list."""
        vdb = _make_vectordb(results=[])
        result = self._retrieve(vdb)
        assert result == []

    # Test 22
    def test_score_below_threshold_is_uncertain(self):
        """A score strictly below CONFIDENCE_THRESHOLD (0.70) must be labelled 'uncertain'."""
        from retrieval import CONFIDENCE_THRESHOLD
        score = CONFIDENCE_THRESHOLD - 0.01  # just under
        doc = _make_doc(metadata={"chunk_id": "c1", "document_name": "guide.pdf", "page_number": 1})
        vdb = _make_vectordb(results=[(doc, score)])
        result = self._retrieve(vdb, k=1)
        assert len(result) == 1
        assert result[0]["confidence"] == "uncertain"
        assert result[0]["score"] == round(score, 3)

    # Test 23
    def test_score_at_exact_threshold_is_confident(self):
        """A score exactly equal to CONFIDENCE_THRESHOLD (0.70) must be labelled 'confident'."""
        from retrieval import CONFIDENCE_THRESHOLD
        score = CONFIDENCE_THRESHOLD  # exactly 0.70
        doc = _make_doc(metadata={"chunk_id": "c1", "document_name": "guide.pdf", "page_number": 1})
        vdb = _make_vectordb(results=[(doc, score)])
        result = self._retrieve(vdb, k=1)
        assert result[0]["confidence"] == "confident"

    # Test 24
    def test_score_above_threshold_is_confident(self):
        """A score strictly above CONFIDENCE_THRESHOLD must be labelled 'confident'."""
        from retrieval import CONFIDENCE_THRESHOLD
        score = CONFIDENCE_THRESHOLD + 0.10
        doc = _make_doc(metadata={"chunk_id": "c1", "document_name": "guide.pdf", "page_number": 1})
        vdb = _make_vectordb(results=[(doc, score)])
        result = self._retrieve(vdb, k=1)
        assert result[0]["confidence"] == "confident"

    # Test 25
    def test_parent_chunk_deduplication_keeps_only_first(self):
        """Two docs sharing the same parent_chunk_id: only the first must appear in output."""
        shared_parent = "parent-001"
        doc1 = _make_doc(
            page_content="child text 1",
            metadata={"chunk_id": "c1", "parent_chunk_id": shared_parent, "document_name": "g.pdf", "page_number": 1},
        )
        doc2 = _make_doc(
            page_content="child text 2",
            metadata={"chunk_id": "c2", "parent_chunk_id": shared_parent, "document_name": "g.pdf", "page_number": 1},
        )
        vdb = _make_vectordb(results=[(doc1, 0.8), (doc2, 0.75)])
        result = self._retrieve(vdb, k=5)
        # Only one result should be returned; the second is deduplicated.
        assert len(result) == 1
        assert result[0]["chunk_id"] == "c1"

    # Test 26
    def test_docs_without_parent_chunk_id_are_not_deduplicated(self):
        """Docs without a parent_chunk_id must both be returned independently."""
        doc1 = _make_doc(
            page_content="text 1",
            metadata={"chunk_id": "c1", "document_name": "g.pdf", "page_number": 1},
        )
        doc2 = _make_doc(
            page_content="text 2",
            metadata={"chunk_id": "c2", "document_name": "g.pdf", "page_number": 2},
        )
        vdb = _make_vectordb(results=[(doc1, 0.8), (doc2, 0.75)])
        result = self._retrieve(vdb, k=5)
        assert len(result) == 2

    # Test 27
    def test_result_count_capped_at_k(self):
        """Output must not exceed k items even if the vector store returns more."""
        docs = [
            _make_doc(
                page_content=f"text {i}",
                metadata={"chunk_id": f"c{i}", "document_name": "g.pdf", "page_number": i},
            )
            for i in range(10)
        ]
        results_pairs = [(d, 0.9) for d in docs]
        vdb = _make_vectordb(results=results_pairs)
        k = 3
        result = self._retrieve(vdb, k=k)
        assert len(result) == k

    # Test 28
    def test_parent_text_takes_precedence_over_page_content(self):
        """When metadata contains 'parent_text', it must be used as the result 'text' field."""
        parent_content = "Full parent section text."
        child_content = "Short child snippet."
        doc = _make_doc(
            page_content=child_content,
            metadata={
                "chunk_id": "c1",
                "document_name": "g.pdf",
                "page_number": 1,
                "parent_text": parent_content,
            },
        )
        vdb = _make_vectordb(results=[(doc, 0.8)])
        result = self._retrieve(vdb, k=1)
        assert result[0]["text"] == parent_content

    # Test 29
    def test_missing_metadata_fields_use_defaults(self):
        """Missing document_name, page_number, and section must fall back to 'unknown', '?', None."""
        doc = _make_doc(page_content="bare text", metadata={})
        vdb = _make_vectordb(results=[(doc, 0.5)])
        result = self._retrieve(vdb, k=1)
        assert result[0]["document_name"] == "unknown"
        assert result[0]["page_number"] == "?"
        assert result[0]["chunk_id"] == "unknown"
        assert result[0]["section"] is None

    # Test 30
    def test_score_is_rounded_to_3_decimal_places(self):
        """Scores must be rounded to exactly 3 decimal places in the output."""
        raw_score = 0.123456789
        doc = _make_doc(metadata={"chunk_id": "c1", "document_name": "g.pdf", "page_number": 1})
        vdb = _make_vectordb(results=[(doc, raw_score)])
        result = self._retrieve(vdb, k=1)
        assert result[0]["score"] == round(raw_score, 3)
        # Confirm it is NOT the raw float (should be 0.123, not 0.123456789)
        assert result[0]["score"] == 0.123


# ===========================================================================
# Section 4 — rrf_merge (hybrid_retriever.py)
# ===========================================================================


@pytest.fixture()
def retriever():
    """Provide a HybridClinicalRetriever instance with no real __init__ side-effects."""
    from hybrid_retriever import HybridClinicalRetriever
    return HybridClinicalRetriever.__new__(HybridClinicalRetriever)


def _chunk(chunk_id: str, extra: dict | None = None) -> tuple:
    """Return a (meta, text, score) tuple suitable for rrf_merge."""
    meta = {"chunk_id": chunk_id, **(extra or {})}
    return meta, f"text for {chunk_id}", 0.0


class TestRrfMerge:
    """Unit tests for HybridClinicalRetriever.rrf_merge."""

    # Test 31
    def test_empty_inputs_return_empty_list(self, retriever):
        """Both inputs empty must produce an empty output list."""
        result = retriever.rrf_merge([], [])
        assert result == []

    # Test 32
    def test_dense_only_returns_items_with_rrf_score(self, retriever):
        """Dense-only input must produce correct output with rrf_score populated."""
        dense = [_chunk("d1"), _chunk("d2")]
        result = retriever.rrf_merge(dense, [])
        assert len(result) == 2
        chunk_ids = {m["chunk_id"] for m, _, _ in result}
        assert chunk_ids == {"d1", "d2"}
        for meta, _, rrf_s in result:
            assert "rrf_score" in meta
            assert isinstance(meta["rrf_score"], float)

    # Test 33
    def test_bm25_only_returns_items_with_rrf_score(self, retriever):
        """BM25-only input must produce correct output with rrf_score populated."""
        bm25 = [_chunk("b1"), _chunk("b2")]
        result = retriever.rrf_merge([], bm25)
        assert len(result) == 2
        for meta, _, _ in result:
            assert "rrf_score" in meta

    # Test 34
    def test_same_chunk_in_both_scores_are_summed(self, retriever):
        """A chunk appearing in both lists must have a higher rrf_score than from either alone."""
        shared_id = "shared-chunk"
        dense = [_chunk(shared_id)]
        bm25 = [_chunk(shared_id)]
        merged = retriever.rrf_merge(dense, bm25)
        # Score from both lists (rank=1 in each)
        k_rrf = 60
        expected = (1.0 / (k_rrf + 1)) + (1.0 / (k_rrf + 1))
        assert len(merged) == 1
        actual_score = merged[0][0]["rrf_score"]
        assert abs(actual_score - round(expected, 6)) < 1e-9

    # Test 35
    def test_top_n_limits_output_count(self, retriever):
        """top_n must cap the number of returned items correctly."""
        dense = [_chunk(f"d{i}") for i in range(10)]
        result = retriever.rrf_merge(dense, [], top_n=3)
        assert len(result) == 3

    # Test 36
    def test_zero_dense_weight_only_bm25_contributes(self, retriever):
        """With dense_weight=0, chunks present only in bm25 must score higher than dense-only ones."""
        dense_only = _chunk("dense-only")
        bm25_only = _chunk("bm25-only")
        result = retriever.rrf_merge(
            [dense_only], [bm25_only], top_n=2, dense_weight=0.0, sparse_weight=1.0
        )
        scores = {m["chunk_id"]: m["rrf_score"] for m, _, _ in result}
        # Dense-only chunk scores 0, BM25-only chunk scores > 0.
        assert scores.get("bm25-only", 0.0) > scores.get("dense-only", 0.0)

    # Test 37
    def test_zero_sparse_weight_only_dense_contributes(self, retriever):
        """With sparse_weight=0, chunks present only in dense must score higher than bm25-only ones."""
        dense_only = _chunk("dense-only")
        bm25_only = _chunk("bm25-only")
        result = retriever.rrf_merge(
            [dense_only], [bm25_only], top_n=2, dense_weight=1.0, sparse_weight=0.0
        )
        scores = {m["chunk_id"]: m["rrf_score"] for m, _, _ in result}
        assert scores.get("dense-only", 0.0) > scores.get("bm25-only", 0.0)

    # Test 38
    def test_larger_k_rrf_produces_smaller_score_spread(self, retriever):
        """A larger k_rrf denominator must reduce the absolute score difference between ranks."""
        dense = [_chunk("first"), _chunk("second")]

        result_small_k = retriever.rrf_merge(dense, [], k_rrf=1)
        result_large_k = retriever.rrf_merge(dense, [], k_rrf=1000)

        scores_small = sorted([m["rrf_score"] for m, _, _ in result_small_k], reverse=True)
        scores_large = sorted([m["rrf_score"] for m, _, _ in result_large_k], reverse=True)

        spread_small = scores_small[0] - scores_small[1]
        spread_large = scores_large[0] - scores_large[1]
        assert spread_small > spread_large, (
            f"Expected larger spread with small k_rrf={spread_small:.6f} "
            f"vs large k_rrf={spread_large:.6f}"
        )

    # Test 39
    def test_rrf_score_field_present_in_all_output_tuples(self, retriever):
        """Every output tuple's metadata dict must contain the 'rrf_score' key."""
        items = [_chunk(f"c{i}") for i in range(5)]
        result = retriever.rrf_merge(items, items[:2])
        for meta, _, _ in result:
            assert "rrf_score" in meta, f"rrf_score missing from {meta}"

    # Test 40
    def test_dense_metadata_preserved_when_chunk_in_both_lists(self, retriever):
        """When a chunk appears in both, the original dense metadata must be preserved."""
        dense_meta = {"chunk_id": "overlap", "dense_score": 0.85, "section_title": "Hypertension"}
        bm25_meta = {"chunk_id": "overlap", "bm25_score": 3.2}
        dense = [(dense_meta, "text", 0.85)]
        bm25 = [(bm25_meta, "text", 3.2)]
        result = retriever.rrf_merge(dense, bm25)
        assert len(result) == 1
        merged_meta = result[0][0]
        # Dense metadata fields must be present.
        assert merged_meta.get("dense_score") == 0.85
        assert merged_meta.get("section_title") == "Hypertension"


# ===========================================================================
# Section 5 — Flask API edge-case tests
# ===========================================================================


@pytest.fixture()
def api_client():
    """Flask test client with a stubbed vectordb and stubbed answer_question."""
    import app as app_module

    app_module.app.config["TESTING"] = True

    stub_answer = {
        "status": "grounded",
        "answer": "Test answer.",
        "evidence": ["Evidence text."],
        "citations": [{"document_name": "guide.pdf", "page_number": 1, "chunk_id": "c1"}],
        "confidence": "confident",
        "top_score": 0.85,
    }

    mock_vdb = _make_vectordb()

    with (
        patch.object(app_module, "get_vectordb", return_value=mock_vdb),
        patch.object(app_module, "answer_question", return_value=stub_answer),
        patch.object(app_module, "_API_AUTH_TOKEN", ""),  # disable auth for most tests
    ):
        with app_module.app.test_client() as client:
            yield client


class TestFlaskApiEdgeCases:
    """Integration-level edge-case tests against the Flask test client."""

    # Test 41
    def test_ask_non_json_body_returns_400(self, api_client):
        """A non-JSON body must be rejected with HTTP 400."""
        resp = api_client.post(
            "/api/ask",
            data="not json at all",
            content_type="text/plain",
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    # Test 42
    def test_ask_empty_json_object_returns_400(self, api_client):
        """An empty JSON object {} (missing 'question') must be rejected with HTTP 400."""
        resp = api_client.post("/api/ask", json={})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    # Test 43
    def test_ask_empty_string_question_returns_400(self, api_client):
        """question='' (empty string) must be rejected with HTTP 400."""
        resp = api_client.post("/api/ask", json={"question": ""})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    # Test 44
    def test_ask_question_over_1000_chars_returns_400(self, api_client):
        """A question exceeding 1000 characters must be rejected with HTTP 400."""
        long_q = "x" * 1001
        resp = api_client.post("/api/ask", json={"question": long_q})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    # Test 45
    def test_ask_k_zero_returns_400(self, api_client):
        """k=0 must be rejected with HTTP 400."""
        resp = api_client.post("/api/ask", json={"question": "valid question", "k": 0})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    # Test 46
    def test_ask_k_21_returns_400(self, api_client):
        """k=21 (above maximum of 20) must be rejected with HTTP 400."""
        resp = api_client.post("/api/ask", json={"question": "valid question", "k": 21})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    # Test 47
    def test_ask_k_true_boolean_returns_400(self, api_client):
        """k=true (JSON boolean) must be rejected with HTTP 400 — bool is not a valid int here."""
        resp = api_client.post("/api/ask", json={"question": "valid question", "k": True})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert "integer" in data["error"].lower()

    # Test 48
    def test_ask_use_reranker_string_returns_400(self, api_client):
        """use_reranker='yes' (string instead of bool) must be rejected with HTTP 400."""
        resp = api_client.post(
            "/api/ask",
            json={"question": "valid question", "use_reranker": "yes"},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert "boolean" in data["error"].lower()

    # Test 49
    def test_live_returns_200_without_loading_index(self):
        """GET /api/live must return 200 {'status': 'ok'} without touching the vector index."""
        import app as app_module

        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as client:
            # Arrange: if get_vectordb is called the test fails.
            with patch.object(
                app_module, "get_vectordb", side_effect=AssertionError("index must NOT be loaded")
            ):
                resp = client.get("/api/live")
        assert resp.status_code == 200
        assert resp.get_json() == {"status": "ok"}

    # Test 50
    def test_health_returns_503_when_vectordb_raises(self):
        """GET /api/health must return HTTP 503 when the vector index raises an exception."""
        import app as app_module

        app_module.app.config["TESTING"] = True
        app_module._vectordb = None  # force fresh load attempt
        with app_module.app.test_client() as client:
            with patch.object(
                app_module, "get_vectordb", side_effect=RuntimeError("ChromaDB unavailable")
            ):
                resp = client.get("/api/health")
        assert resp.status_code == 503
        data = resp.get_json()
        assert data.get("ready") is False


# ===========================================================================
# Section 6 — _parse_request_options unit tests (pure function, no HTTP)
# ===========================================================================


class TestParseRequestOptions:
    """Direct tests of the _parse_request_options helper to cover all validation branches."""

    def _parse(self, body):
        from app import _parse_request_options
        return _parse_request_options(body)

    def test_non_dict_body_returns_error(self):
        """A non-dict body (e.g. a list) must return an error message."""
        opts, err = self._parse([1, 2, 3])
        assert opts is None
        assert "JSON object" in err

    def test_valid_body_parses_correctly(self):
        """A fully valid body must be parsed without error."""
        opts, err = self._parse({"question": "What is BP?", "k": 3, "use_reranker": False})
        assert err is None
        question, k, use_reranker = opts
        assert question == "What is BP?"
        assert k == 3
        assert use_reranker is False

    def test_question_whitespace_only_returns_error(self):
        """A question consisting solely of whitespace must be rejected."""
        opts, err = self._parse({"question": "   "})
        assert opts is None
        assert err is not None

    def test_k_bool_true_returns_error(self):
        """k=True must be rejected because bool is a subclass of int."""
        opts, err = self._parse({"question": "valid", "k": True})
        assert opts is None
        assert "integer" in err.lower()

    def test_k_bool_false_returns_error(self):
        """k=False must be rejected (same reason as True)."""
        opts, err = self._parse({"question": "valid", "k": False})
        assert opts is None
        assert "integer" in err.lower()

    def test_k_exactly_1_is_valid(self):
        """k=1 (lower boundary) must be accepted."""
        opts, err = self._parse({"question": "valid", "k": 1})
        assert err is None

    def test_k_exactly_20_is_valid(self):
        """k=20 (upper boundary) must be accepted."""
        opts, err = self._parse({"question": "valid", "k": 20})
        assert err is None

    def test_use_reranker_non_bool_returns_error(self):
        """use_reranker=1 (integer, not bool) must be rejected."""
        opts, err = self._parse({"question": "valid", "use_reranker": 1})
        assert opts is None
        assert "boolean" in err.lower()
