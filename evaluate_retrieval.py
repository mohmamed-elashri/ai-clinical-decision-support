"""
evaluate_retrieval.py — Day 2 Module 1
Measures Precision@K (official), PageRecall@K, DocHit@K, and out-of-scope controls.
"""
import time

import config
from evaluation_set import EVAL_SET, NEGATIVE_EVAL_SET, POSITIVE_EVAL_SET
from ingest import get_embedding_function
from langchain_chroma import Chroma


def _page_hits(results, item) -> int:
    return sum(
        1
        for doc, _score in results
        if doc.metadata.get("document_name") == item["expected_document"]
        and doc.metadata.get("page_number") in item["expected_pages"]
    )


def _doc_hit(results, item) -> bool:
    return any(
        doc.metadata.get("document_name") == item["expected_document"]
        for doc, _score in results
    )


def evaluate_k_values(vectordb, eval_set=None, k_values=None):
    """
    Evaluate retrieval metrics.

    PageRecall@K  — fraction of queries where at least one expected *page* appears in top-K.
    DocHit@K      — fraction of queries where the expected *document* appears in top-K.
    Precision@K   — average (relevant pages in top-K) / K across positive queries.
    """
    eval_set = POSITIVE_EVAL_SET if eval_set is None else eval_set
    k_values = k_values or config.EVAL_K_VALUES
    summary = []

    for k in k_values:
        precisions, page_recalls, doc_hits, latencies = [], [], [], []

        for item in eval_set:
            t0 = time.perf_counter()
            results = vectordb.similarity_search_with_relevance_scores(
                item["question"], k=k
            )
            latencies.append(time.perf_counter() - t0)

            page_hits = _page_hits(results, item)
            precisions.append(page_hits / k)
            page_recalls.append(1 if page_hits > 0 else 0)
            doc_hits.append(1 if _doc_hit(results, item) else 0)

        n = len(eval_set)
        summary.append(
            {
                "k": k,
                "precision": sum(precisions) / n,
                "page_recall": sum(page_recalls) / n,
                "doc_hit": sum(doc_hits) / n,
                "latency_ms": 1000 * sum(latencies) / n,
            }
        )

    return summary


def evaluate_negative_controls(vectordb, negative_set=None, k=None):
    """Out-of-scope queries should stay below the confidence threshold."""
    negative_set = negative_set or NEGATIVE_EVAL_SET
    k = k or config.TOP_K
    results = []

    for item in negative_set:
        hits = vectordb.similarity_search_with_relevance_scores(item["question"], k=k)
        top_score = hits[0][1] if hits else 0.0
        max_allowed = item.get("max_top_score", config.OUT_OF_SCOPE_MAX_SCORE)
        passed = top_score < max_allowed
        results.append(
            {
                "question": item["question"],
                "top_score": top_score,
                "max_allowed": max_allowed,
                "passed": passed,
            }
        )

    return results


if __name__ == "__main__":
    print("Loading vectorstore...")
    t0 = time.time()
    vectordb = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=str(config.VECTOR_DB_DIR),
    )
    chunk_count = vectordb._collection.count()
    print(f"  Loaded in {time.time() - t0:.1f}s | Total chunks: {chunk_count}\n")

    print("=" * 88)
    print(
        f"{'K':>4} | {'Precision@K':>12} | {'PageRecall@K':>13} | "
        f"{'DocHit@K':>10} | {'Latency':>10} | Decision"
    )
    print("-" * 88)

    k_metrics = evaluate_k_values(vectordb)
    optimal = next(m for m in k_metrics if m["k"] == config.TOP_K)

    for m in k_metrics:
        k = m["k"]
        if k == config.TOP_K:
            flag = " ← SELECTED"
        elif k == 4:
            flag = " ← hackathon start point"
        else:
            flag = ""
        print(
            f"{k:>4} | {m['precision']:>12.2%} | {m['page_recall']:>13.2%} | "
            f"{m['doc_hit']:>10.2%} | {m['latency_ms']:>8.1f}ms |{flag}"
        )

    print("=" * 88)
    print(
        f"\nSelected K = {config.TOP_K}  "
        f"(Precision@{config.TOP_K} = {optimal['precision']:.2%}, "
        f"PageRecall@{config.TOP_K} = {optimal['page_recall']:.2%}, "
        f"DocHit@{config.TOP_K} = {optimal['doc_hit']:.2%})"
    )

    print(f"\n--- Per-question breakdown at K={config.TOP_K} (positive set) ---")
    for item in POSITIVE_EVAL_SET:
        results = vectordb.similarity_search_with_relevance_scores(
            item["question"], k=config.TOP_K
        )
        page_hits = _page_hits(results, item)
        doc_hit = _doc_hit(results, item)
        top_score = results[0][1] if results else 0
        page_marker = "✅" if page_hits > 0 else "❌"
        doc_marker = "📄" if doc_hit else "  "
        print(
            f"  {page_marker}{doc_marker} [{item['difficulty']:>6}] "
            f"P={page_hits / config.TOP_K:.0%} | top={top_score:.3f} | "
            f"{item['question'][:55]}"
        )

    print(f"\n--- Out-of-scope negative controls at K={config.TOP_K} ---")
    neg_results = evaluate_negative_controls(vectordb)
    for r in neg_results:
        marker = "✅" if r["passed"] else "❌"
        print(
            f"  {marker} top={r['top_score']:.3f} (max {r['max_allowed']:.2f}) | "
            f"{r['question'][:60]}"
        )
