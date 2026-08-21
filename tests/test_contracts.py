import json
import unittest
from unittest.mock import patch

import generation
from llm_client import verify_response_provenance
from evaluation_set_30 import BENCHMARK_30
from hybrid_retriever import HybridClinicalRetriever, expand_clinical_query
from retrieval import retrieve_with_citation


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "chunk_id": "guide_p4_c1",
                "document_name": "guide.pdf",
                "page_number": 4,
                "text": "Target blood pressure should be below 140/90 mmHg.",
            }
        ]

    def test_live_probe_does_not_load_index(self):
        import app

        with patch.object(app, "get_vectordb", side_effect=AssertionError("index loaded")):
            response = app.app.test_client().get("/api/live")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ok")

    def test_protected_endpoint_rejects_missing_api_key(self):
        import app

        with patch.object(app, "_API_AUTH_TOKEN", "test-token"):
            response = app.app.test_client().post(
                "/api/retrieve", json={"question": "test"}
            )
        self.assertEqual(response.status_code, 401)

    def test_security_headers_and_request_id_are_present(self):
        import app

        response = app.app.test_client().get(
            "/api/live", headers={"X-Request-ID": "not-a-uuid"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertNotEqual(response.headers["X-Request-ID"], "not-a-uuid")

    def test_oversized_request_returns_json_error(self):
        import app

        payload = {"question": "x" * (app.app.config["MAX_CONTENT_LENGTH"] + 1)}
        response = app.app.test_client().post("/api/ask", json=payload)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json["error"], "Request body is too large.")

    def test_matching_citation_and_evidence_are_accepted(self):
        response = {
            "status": "grounded",
            "answer": "Target blood pressure should be below 140/90 mmHg.",
            "evidence": ["Target blood pressure should be below 140/90 mmHg."],
            "citations": [
                {
                    "document_name": "guide.pdf",
                    "page_number": 4,
                    "chunk_id": "guide_p4_c1",
                }
            ],
        }
        self.assertEqual(verify_response_provenance(response, self.chunks), [])

    def test_fabricated_source_is_rejected(self):
        response = {
            "status": "grounded",
            "answer": "Unsupported answer.",
            "evidence": ["Evidence not present in the source."],
            "citations": [
                {
                    "document_name": "fake.pdf",
                    "page_number": 99,
                    "chunk_id": "fake_chunk",
                }
            ],
        }
        errors = verify_response_provenance(response, self.chunks)
        self.assertTrue(any("chunk_id" in error for error in errors))
        self.assertTrue(any("evidence snippet" in error for error in errors))

    def test_abstain_does_not_require_citations(self):
        response = {"status": "abstain", "answer": "", "citations": []}
        self.assertEqual(verify_response_provenance(response, self.chunks), [])


class ScoreContractTests(unittest.TestCase):
    def test_retrieval_rejects_zero_k(self):
        with self.assertRaises(ValueError):
            retrieve_with_citation(object(), "question", k=0)

    def test_parser_rejects_text_outside_json(self):
        from llm_client import parse_and_clean_json

        with self.assertRaises(json.JSONDecodeError):
            parse_and_clean_json('{"status": "abstain"} trailing text')

    def test_benchmark_contains_thirty_questions(self):
        self.assertEqual(len(BENCHMARK_30), 30)

    def test_rrf_preserves_dense_score_when_bm25_repeats_chunk(self):
        retriever = HybridClinicalRetriever.__new__(HybridClinicalRetriever)
        dense = [
            ({"chunk_id": "chunk-1", "dense_score": 0.4}, "text", 0.4)
        ]
        bm25 = [({"chunk_id": "chunk-1"}, "text", 3.0)]
        merged = retriever.rrf_merge(dense, bm25, top_n=1)
        self.assertEqual(merged[0][0]["dense_score"], 0.4)
        self.assertIn("rrf_score", merged[0][0])

    def test_rrf_weights_change_fusion_score(self):
        retriever = HybridClinicalRetriever.__new__(HybridClinicalRetriever)
        dense = [({"chunk_id": "dense", "dense_score": 0.8}, "dense", 0.8)]
        sparse = [({"chunk_id": "sparse"}, "sparse", 3.0)]
        dense_first = retriever.rrf_merge(
            dense, sparse, top_n=2, dense_weight=2.0, sparse_weight=1.0
        )
        sparse_first = retriever.rrf_merge(
            dense, sparse, top_n=2, dense_weight=1.0, sparse_weight=2.0
        )
        self.assertNotEqual(dense_first[0][0]["chunk_id"], sparse_first[0][0]["chunk_id"])

    def test_query_expansion_preserves_original_question(self):
        expanded = expand_clinical_query("What is the BP target for CVD?")
        self.assertIn("What is the BP target for CVD?", expanded)
        self.assertIn("cardiovascular disease", expanded)

    def test_query_expansion_is_opt_in(self):
        retriever = HybridClinicalRetriever.__new__(HybridClinicalRetriever)
        retriever._dense_search = lambda query, top_n: [
            ({"chunk_id": query, "dense_score": 0.8}, query, 0.8)
        ]
        retriever._bm25_search = lambda query, top_n: []
        retriever.rrf_merge = lambda dense, sparse, **kwargs: dense
        result = retriever.retrieve("CVD target", top_k=1, use_reranker=False)
        self.assertEqual(result[0]["chunk_id"], "CVD target")

    def test_abstention_uses_dense_score_over_rerank_score(self):
        retrieved = [
            {
                "score": 9.9,
                "dense_score": 0.20,
                "confidence": "uncertain",
                "text": "Retrieved text.",
                "document_name": "guide.pdf",
                "page_number": 1,
                "chunk_id": "chunk-1",
            }
        ]
        with patch.object(generation, "retrieve_with_citation", return_value=retrieved):
            result = generation.answer_question(object(), "question", use_reranker=True)
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["top_score"], 0.2)


if __name__ == "__main__":
    unittest.main()
