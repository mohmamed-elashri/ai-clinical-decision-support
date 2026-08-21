"""Benchmark Dense, BM25, and RRF retrieval on 30 verified questions.

Run from the project root:
    python benchmark_30.py

The script does not rebuild the index and does not load FlashRank unless the
optional reranker benchmark is explicitly added later. Results are written to
benchmark_30_results.json for reproducibility.
"""

import json
import argparse
import time
from pathlib import Path

import config
from evaluation_set_30 import (
    BENCHMARK_30,
    NEGATIVE_SET,
    PRIMARY_POSITIVE_SET,
    PARAPHRASE_POSITIVE_SET,
)
from ingest import get_embedding_function
from langchain_chroma import Chroma
from hybrid_retriever import HybridClinicalRetriever, expand_clinical_query

RESULT_PATH = Path(__file__).resolve().parent / "benchmark_30_results.json"
K = config.TOP_K


def _dense(vectordb, question):
    return vectordb.similarity_search_with_relevance_scores(question, k=K)


def _bm25(hybrid, question):
    return hybrid._bm25_search(question, top_n=K)


def _rrf_from_cache(
    hybrid, cached, dense_weight=1.0, sparse_weight=1.0, diversify=False
):
    candidates = hybrid.rrf_merge(
        cached["dense"],
        cached["bm25"],
        top_n=15,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )
    if diversify:
        selected = []
        seen_pages = set()
        for candidate in candidates:
            meta = candidate[0]
            page_key = (meta.get("document_name"), meta.get("page_number"))
            if page_key in seen_pages:
                continue
            selected.append(candidate)
            seen_pages.add(page_key)
            if len(selected) == K:
                break
        candidates = selected
    else:
        candidates = candidates[:K]
    return [
        {
            "rank": rank,
            "text": text,
            "document_name": meta.get("document_name"),
            "page_number": meta.get("page_number"),
            "chunk_id": meta.get("chunk_id"),
            "score": round(float(rrf_score), 3),
            "rrf_score": round(float(rrf_score), 6),
            "dense_score": meta.get("dense_score", 0.0),
        }
        for rank, (meta, text, rrf_score) in enumerate(candidates, start=1)
    ]


def _rerank(retriever, question):
    return retriever.retrieve(question, top_k=K)


def _page_match(item, document_name, page_number):
    return (
        item.get("expected_document") == document_name
        and page_number in item.get("expected_pages", [])
    )


def _evaluate_positive(name, search, questions):
    precisions = []
    recalls = []
    doc_hits = []
    latencies = []
    for item in questions:
        started = time.perf_counter()
        results = search(item["question"])
        latencies.append((time.perf_counter() - started) * 1000)
        if name == "dense":
            normalized = [
                (doc.metadata.get("document_name"), doc.metadata.get("page_number"))
                for doc, _score in results
            ]
        elif name == "bm25":
            normalized = [
                (meta.get("document_name"), meta.get("page_number"))
                for meta, _text, _score in results
            ]
        else:
            normalized = [
                (result.get("document_name"), result.get("page_number"))
                for result in results
            ]
        hits = sum(1 for document, page in normalized if _page_match(item, document, page))
        precisions.append(hits / K)
        recalls.append(1 if hits else 0)
        doc_hits.append(
            1
            if any(document == item["expected_document"] for document, _page in normalized)
            else 0
        )
    return {
        "n": len(questions),
        "precision_at_3": sum(precisions) / len(precisions),
        "page_recall_at_3": sum(recalls) / len(recalls),
        "doc_hit_at_3": sum(doc_hits) / len(doc_hits),
        "latency_ms_mean": sum(latencies) / len(latencies),
        "latency_ms_p95": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
    }


def _evaluate_negative(name, search):
    details = []
    for item in NEGATIVE_SET:
        results = search(item["question"])
        limit = item.get("max_top_score", config.OUT_OF_SCOPE_MAX_SCORE)
        top_score = None
        if name == "dense" and results:
            top_score = float(results[0][1])
        elif (name.startswith("rrf_") or name in {"dense_rerank", "hybrid_rerank"}) and results:
            top_score = results[0].get("dense_score")
        details.append(
            {
                "question": item["question"],
                "top_dense_score": top_score,
                "limit": limit,
                "passed": top_score is not None and top_score < limit,
                "score_available": top_score is not None,
            }
        )
    scored = [detail for detail in details if detail["score_available"]]
    return {
        "n": len(details),
        "score_available": len(scored),
        "abstain_pass_rate": (
            sum(detail["passed"] for detail in scored) / len(scored) if scored else None
        ),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-rerank",
        action="store_true",
        help="Also benchmark Dense + FlashRank and Hybrid + FlashRank.",
    )
    args = parser.parse_args()
    vectordb = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=str(config.VECTOR_DB_DIR),
    )
    hybrid = HybridClinicalRetriever(vectordb=vectordb, load_reranker=False)
    all_questions = [item["question"] for item in BENCHMARK_30]
    cached = {
        question: {
            "dense": hybrid._dense_search(question, top_n=15),
            "bm25": hybrid._bm25_search(question, top_n=15),
        }
        for question in all_questions
    }
    expanded_cached = {
        question: {
            "dense": hybrid._dense_search(expand_clinical_query(question), top_n=15),
            "bm25": hybrid._bm25_search(expand_clinical_query(question), top_n=15),
        }
        for question in all_questions
    }
    searches = {
        "dense": lambda question: _dense(vectordb, question),
        "bm25": lambda question: _bm25(hybrid, question),
        "rrf_1.0_1.0": lambda question: _rrf_from_cache(hybrid, cached[question], 1.0, 1.0),
        "rrf_1.5_1.0": lambda question: _rrf_from_cache(hybrid, cached[question], 1.5, 1.0),
        "rrf_2.0_1.0": lambda question: _rrf_from_cache(hybrid, cached[question], 2.0, 1.0),
        "rrf_1.0_1.5": lambda question: _rrf_from_cache(hybrid, cached[question], 1.0, 1.5),
        "rrf_expanded": lambda question: _rrf_from_cache(
            hybrid, expanded_cached[question], 1.0, 1.0
        ),
        "rrf_diverse": lambda question: _rrf_from_cache(
            hybrid, cached[question], 1.0, 1.0, diversify=True
        ),
    }
    rerankers = {}
    if args.with_rerank:
        from dense_reranker import DenseRerankRetriever

        rerankers["dense_rerank"] = DenseRerankRetriever(vectordb=vectordb)
        rerankers["hybrid_rerank"] = HybridClinicalRetriever(vectordb=vectordb)
        searches["dense_rerank"] = lambda question: _rerank(
            rerankers["dense_rerank"], question
        )
        searches["hybrid_rerank"] = lambda question: rerankers["hybrid_rerank"].retrieve(
            question, top_k=K, fetch_candidates=15, use_reranker=True
        )

    output = {
        "benchmark": "30-question retrieval benchmark",
        "corpus_chunks": vectordb._collection.count(),
        "k": K,
        "sets": {
            "primary": len(PRIMARY_POSITIVE_SET),
            "paraphrase": len(PARAPHRASE_POSITIVE_SET),
            "negative": len(NEGATIVE_SET),
        },
        "architectures": {},
    }
    for name, search in searches.items():
        output["architectures"][name] = {
            "primary": _evaluate_positive(name, search, PRIMARY_POSITIVE_SET),
            "paraphrase": _evaluate_positive(name, search, PARAPHRASE_POSITIVE_SET),
            "negative": _evaluate_negative(name, search),
        }

    RESULT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
