"""
evaluate_embeddings.py — Day 2 Module 3
Compare embedding models on the same chunk config and evaluation set.
"""
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config
from evaluation_set import POSITIVE_EVAL_SET
from ingest import chunk_documents, get_embedding_function, load_pdfs
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings


def _precision_at_k(vectordb, k: int) -> float:
    scores = []
    for item in POSITIVE_EVAL_SET:
        results = vectordb.similarity_search_with_relevance_scores(
            item["question"], k=k
        )
        hits = sum(
            1
            for doc, _ in results
            if doc.metadata.get("document_name") == item["expected_document"]
            and doc.metadata.get("page_number") in item["expected_pages"]
        )
        scores.append(hits / k)
    return sum(scores) / len(scores)


def _page_recall_at_k(vectordb, k: int) -> float:
    hits = 0
    for item in POSITIVE_EVAL_SET:
        results = vectordb.similarity_search_with_relevance_scores(
            item["question"], k=k
        )
        if any(
            doc.metadata.get("document_name") == item["expected_document"]
            and doc.metadata.get("page_number") in item["expected_pages"]
            for doc, _ in results
        ):
            hits += 1
    return hits / len(POSITIVE_EVAL_SET)


def benchmark_model(model_name: str, chunks, k: int | None = None):
    """Build a transient index and measure Precision@K + latency."""
    k = k or config.TOP_K
    embed = FastEmbedEmbeddings(model_name=model_name)
    collection = f"bench_{model_name.replace('/', '_').replace('.', '_')}"

    vdb = Chroma.from_documents(
        documents=chunks,
        embedding=embed,
        collection_name=collection,
        ids=[c.metadata["chunk_id"] for c in chunks],
    )

    latencies = []
    for item in POSITIVE_EVAL_SET:
        t0 = time.perf_counter()
        vdb.similarity_search_with_relevance_scores(item["question"], k=k)
        latencies.append(time.perf_counter() - t0)

    metrics = {
        "model": model_name,
        "precision": _precision_at_k(vdb, k),
        "page_recall": _page_recall_at_k(vdb, k),
        "latency_ms": 1000 * sum(latencies) / len(latencies),
    }
    vdb.delete_collection()
    return metrics


def run_embedding_benchmark(models=None, k: int | None = None):
    models = models or config.EMBEDDING_BENCHMARK_MODELS
    k = k or config.TOP_K

    print("Building shared chunk set for fair embedding comparison...")
    pages = load_pdfs(config.DATA_DIR)
    chunks = chunk_documents(pages)
    print(f"  {len(pages)} pages → {len(chunks)} chunks\n")

    print("=" * 90)
    print(
        f"{'Model':<45} | {'Precision@K':>12} | {'PageRecall@K':>13} | {'Latency':>10}"
    )
    print("-" * 90)

    results = []
    for model_name in models:
        print(f"  Benchmarking {model_name}...", flush=True)
        try:
            m = benchmark_model(model_name, chunks, k=k)
            results.append(m)
            print(
                f"{m['model']:<45} | {m['precision']:>12.2%} | "
                f"{m['page_recall']:>13.2%} | {m['latency_ms']:>8.1f}ms"
            )
        except Exception as exc:
            print(f"  SKIP  {model_name}: {exc}")

    print("=" * 90)
    if not results:
        raise RuntimeError("No embedding models could be benchmarked.")

    best = max(results, key=lambda x: (x["precision"], -x["latency_ms"]))
    print(
        f"\nRecommended model: {best['model']} "
        f"(Precision@{k}={best['precision']:.1%}, latency={best['latency_ms']:.1f}ms)"
    )
    if len(results) < len(models):
        print(
            f"Note: {len(models) - len(results)} model(s) skipped "
            "(download/cache unavailable — re-run when online)."
        )
    return results


if __name__ == "__main__":
    run_embedding_benchmark()
