"""
evaluate_robustness.py — Anti-overfitting validation suite

Proves retrieval generalizes beyond literal PDF phrasing:
  1. Paraphrase set (10 queries, held-out wording)
  2. Expanded hard-negative controls (6 queries)
  3. Side-by-side comparison with primary EVAL_SET
"""
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from typing import List

import config
from dense_reranker import DenseRerankRetriever
from evaluate_retrieval import _doc_hit, _page_hits
from evaluation_set import NEGATIVE_EVAL_SET, POSITIVE_EVAL_SET
from evaluation_set_robustness import HARD_NEGATIVE_SET, PARAPHRASE_EVAL_SET
from ingest import get_embedding_function
from langchain_chroma import Chroma


def _metrics_for_set(vectordb, eval_set, k=None) -> dict:
    k = k or config.TOP_K
    precisions, page_recalls, doc_hits, latencies = [], [], [], []

    for item in eval_set:
        t0 = time.perf_counter()
        results = vectordb.similarity_search_with_relevance_scores(item["question"], k=k)
        latencies.append(time.perf_counter() - t0)

        hits = _page_hits(results, item)
        precisions.append(hits / k)
        page_recalls.append(1 if hits > 0 else 0)
        doc_hits.append(1 if _doc_hit(results, item) else 0)

    n = len(eval_set)
    return {
        "n": n,
        "precision": sum(precisions) / n,
        "page_recall": sum(page_recalls) / n,
        "doc_hit": sum(doc_hits) / n,
        "latency_ms": 1000 * sum(latencies) / n,
    }


def _negative_pass_details(vectordb, negative_set, k=None) -> List[dict]:
    """
    Run negative-control queries and return per-item details.

    Returns a list of dicts with keys:
        question, top_score, limit, passed

    Separating data collection from display eliminates the double-query
    pattern that was in the original code (queries were run once for the
    aggregate pass_rate and again for the per-item printout).
    """
    k = k or config.TOP_K
    details = []
    for item in negative_set:
        hits = vectordb.similarity_search_with_relevance_scores(item["question"], k=k)
        top_score = hits[0][1] if hits else 0.0
        limit = item.get("max_top_score", config.OUT_OF_SCOPE_MAX_SCORE)
        details.append(
            {
                "question": item["question"],
                "top_score": top_score,
                "limit": limit,
                "passed": top_score < limit,
            }
        )
    return details


def _reranker_metrics(retriever: DenseRerankRetriever, eval_set, k=None) -> dict:
    k = k or config.TOP_K
    precisions, page_recalls, doc_hits = [], [], []

    for item in eval_set:
        res = retriever.retrieve(item["question"], top_k=k)
        hits = sum(
            1
            for r in res
            if r["document_name"] == item["expected_document"]
            and r["page_number"] in item["expected_pages"]
        )
        precisions.append(hits / k)
        page_recalls.append(1 if hits > 0 else 0)
        doc_hits.append(
            1 if any(r["document_name"] == item["expected_document"] for r in res) else 0
        )

    n = len(eval_set)
    return {
        "precision": sum(precisions) / n,
        "page_recall": sum(page_recalls) / n,
        "doc_hit": sum(doc_hits) / n,
    }


def main():
    print("=" * 88)
    print(" ROBUSTNESS & ANTI-OVERFITTING VALIDATION")
    print("=" * 88)

    vectordb = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=str(config.VECTOR_DB_DIR),
    )
    chunk_count = vectordb._collection.count()
    print(f" Corpus: {chunk_count} chunks | Primary eval: held separate from paraphrase set\n")

    primary = _metrics_for_set(vectordb, POSITIVE_EVAL_SET)
    paraphrase = _metrics_for_set(vectordb, PARAPHRASE_EVAL_SET)

    # Collect negative-control details once — reuse for both aggregate and
    # per-item display (eliminates double queries from the original code).
    all_negative_details = _negative_pass_details(
        vectordb, NEGATIVE_EVAL_SET + HARD_NEGATIVE_SET
    )
    neg_rate = sum(1 for d in all_negative_details if d["passed"]) / len(all_negative_details)

    print(f"\n--- Primary EVAL_SET (n={primary['n']}) — development benchmark ---")
    print(
        f"  Precision@{config.TOP_K}={primary['precision']:.1%} | "
        f"PageRecall@{config.TOP_K}={primary['page_recall']:.1%} | "
        f"DocHit@{config.TOP_K}={primary['doc_hit']:.1%} | "
        f"latency={primary['latency_ms']:.1f}ms"
    )

    print(f"\n--- Paraphrase held-out set (n={paraphrase['n']}) — anti-leakage test ---")
    print(
        f"  Precision@{config.TOP_K}={paraphrase['precision']:.1%} | "
        f"PageRecall@{config.TOP_K}={paraphrase['page_recall']:.1%} | "
        f"DocHit@{config.TOP_K}={paraphrase['doc_hit']:.1%}"
    )
    gap = primary["page_recall"] - paraphrase["page_recall"]
    print(f"  Recall gap (primary − paraphrase): {gap:.1%}")
    if paraphrase["page_recall"] >= 0.85:
        print("  ✅ Paraphrase PageRecall ≥ 85% — low literal-match overfitting risk")
    elif paraphrase["page_recall"] >= 0.70:
        print("  ⚠️  Paraphrase PageRecall 70–85% — acceptable but monitor wording sensitivity")
    else:
        print("  ❌ Paraphrase PageRecall < 70% — possible overfitting to PDF phrasing")

    print(f"\n--- Negative controls (n={len(all_negative_details)}) ---")
    print(f"  Abstain pass rate (top score below threshold): {neg_rate:.1%}")
    for d in all_negative_details:
        mark = "✅" if d["passed"] else "❌"
        print(f"  {mark} top={d['top_score']:.3f} (max {d['limit']:.2f}) | {d['question'][:55]}")

    print("\n--- Dense + Re-ranker (fetch 10 → top 3) on paraphrase set ---")
    try:
        reranker = DenseRerankRetriever(vectordb=vectordb)
        rerank_m = _reranker_metrics(reranker, PARAPHRASE_EVAL_SET)
        print(
            f"  Precision@{config.TOP_K}={rerank_m['precision']:.1%} | "
            f"PageRecall@{config.TOP_K}={rerank_m['page_recall']:.1%} | "
            f"DocHit@{config.TOP_K}={rerank_m['doc_hit']:.1%}"
        )
    except Exception as exc:
        print(f"  SKIP  Dense re-ranker unavailable: {exc}")

    print("\n--- Per-paraphrase breakdown ---")
    for item in PARAPHRASE_EVAL_SET:
        results = vectordb.similarity_search_with_relevance_scores(
            item["question"], k=config.TOP_K
        )
        hits = _page_hits(results, item)
        mark = "✅" if hits > 0 else "❌"
        print(f"  {mark} P={hits / config.TOP_K:.0%} | {item['question'][:60]}")

    print("\n" + "=" * 88)
    print(
        " Interpretation: 100% on primary set reflects tuning on a small local benchmark "
        f"({primary['n']} queries, {chunk_count} chunks). Paraphrase recall is the generalization signal."
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
