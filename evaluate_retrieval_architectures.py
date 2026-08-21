"""
evaluate_retrieval_architectures.py — Empirical comparison of retrieval architectures
1. Dense Vector Search (bge-small)
2. Sparse Keyword Search (BM25)
3. Hybrid RRF (Dense + BM25)
4. Hybrid + Reranker (TinyBERT + BM25 — legacy)
5. Dense + Re-ranker (MiniLM — recommended rerank path)
"""

import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config
from dense_reranker import DenseRerankRetriever
from evaluation_set import POSITIVE_EVAL_SET
from ingest import get_embedding_function
from langchain_chroma import Chroma
from hybrid_retriever import HybridClinicalRetriever

def run_architectural_benchmark():
    vectordb = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=str(config.VECTOR_DB_DIR),
    )
    hybrid = HybridClinicalRetriever(vectordb=vectordb)
    dense_rerank = DenseRerankRetriever(vectordb=vectordb)

    architectures = {
        "1. Dense Vector (bge-small)": lambda q: [
            {
                "document_name": doc.metadata.get("document_name"),
                "page_number": doc.metadata.get("page_number"),
            }
            for doc, _s in vectordb.similarity_search_with_relevance_scores(q, k=config.TOP_K)
        ],
        "2. Sparse Keyword (BM25)": lambda q: [
            {
                "document_name": meta.get("document_name"),
                "page_number": meta.get("page_number"),
            }
            for meta, text, score in hybrid._bm25_search(q, top_n=config.TOP_K)
        ],
        "3. Hybrid RRF (Dense + BM25)": lambda q: [
            {
                "document_name": r["document_name"],
                "page_number": r["page_number"],
            }
            for r in hybrid.retrieve(q, top_k=config.TOP_K, use_reranker=False)
        ],
        "4. Hybrid + Reranker (TinyBERT)": lambda q: [
            {
                "document_name": r["document_name"],
                "page_number": r["page_number"],
            }
            for r in hybrid.retrieve(q, top_k=config.TOP_K, use_reranker=True)
        ],
        "5. Dense + Re-ranker (MiniLM)": lambda q: [
            {
                "document_name": r["document_name"],
                "page_number": r["page_number"],
            }
            for r in dense_rerank.retrieve(q, top_k=config.TOP_K)
        ],
    }

    print("=" * 88)
    print(f" RETRIEVAL ARCHITECTURE BENCHMARK (n={len(POSITIVE_EVAL_SET)}, K={config.TOP_K})")
    print("=" * 88)
    print(f"{'Architecture':<34} | {'P@1':>8} | {'P@3':>8} | {'PageRecall@3':>13} | {'Latency':>10}")
    print("-" * 88)

    benchmark_summary = []

    for name, fn in architectures.items():
        t0 = time.time()
        p1_list, p3_list, page_recall_list = [], [], []

        for item in POSITIVE_EVAL_SET:
            res = fn(item["question"])

            p1 = 1 if (
                res
                and res[0]["document_name"] == item["expected_document"]
                and res[0]["page_number"] in item["expected_pages"]
            ) else 0
            p1_list.append(p1)

            hits = sum(
                1 for r in res
                if r["document_name"] == item["expected_document"]
                and r["page_number"] in item["expected_pages"]
            )
            p3_list.append(hits / float(config.TOP_K))
            page_recall_list.append(1 if hits > 0 else 0)

        latency = (time.time() - t0) / len(POSITIVE_EVAL_SET) * 1000
        avg_p1 = sum(p1_list) / len(POSITIVE_EVAL_SET)
        avg_p3 = sum(p3_list) / len(POSITIVE_EVAL_SET)
        avg_pr = sum(page_recall_list) / len(POSITIVE_EVAL_SET)

        benchmark_summary.append({
            "name": name,
            "p1": avg_p1,
            "p3": avg_p3,
            "page_recall": avg_pr,
            "latency": latency,
        })

        print(f"{name:<34} | {avg_p1:>7.1%} | {avg_p3:>7.1%} | {avg_pr:>12.1%} | {latency:>8.1f}ms")

    print("=" * 88)
    best = max(benchmark_summary, key=lambda x: (x["p3"], x["p1"], -x["latency"]))
    print(f"\nRecommended: {best['name']}")
    print(
        f"   P@3={best['p3']:.1%} | PageRecall@3={best['page_recall']:.1%} | "
        f"Latency={best['latency']:.1f}ms\n"
    )

if __name__ == "__main__":
    run_architectural_benchmark()
