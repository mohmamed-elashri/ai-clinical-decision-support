"""Day 2 Definition of Done — automated verification."""
import importlib.util
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config
from evaluate_retrieval import evaluate_k_values, evaluate_negative_controls
from evaluation_set import EVAL_SET, NEGATIVE_EVAL_SET, POSITIVE_EVAL_SET
from ingest import chunk_documents, get_embedding_function, load_pdfs
from langchain_chroma import Chroma
from retrieval import retrieve_with_citation

ROOT = Path(__file__).resolve().parent
FAILURES: list[str] = []
CHECKS: list[str] = []


def ok(msg: str) -> None:
    CHECKS.append(f"PASS  {msg}")


def fail(msg: str) -> None:
    FAILURES.append(f"FAIL  {msg}")


def main() -> int:
    # --- Required files ---
    required = [
        "evaluate_retrieval.py",
        "chunk_ablation.py",
        "retrieval.py",
        "evaluate_embeddings.py",
        "evaluate_retrieval_architectures.py",
        "evaluate_robustness.py",
        "evaluation_set.py",
        "evaluation_set_robustness.py",
        "DAY2_REPORT.md",
    ]
    for name in required:
        if (ROOT / name).exists():
            ok(f"{name} present")
        else:
            fail(f"{name} missing")

    if importlib.util.find_spec("evaluate_embeddings") is None:
        fail("evaluate_embeddings module not importable")
    else:
        ok("evaluate_embeddings.py importable")

    if importlib.util.find_spec("evaluate_robustness") is None:
        fail("evaluate_robustness module not importable")
    else:
        ok("evaluate_robustness.py importable (anti-overfitting suite)")

    # --- Evaluation set structure ---
    if len(POSITIVE_EVAL_SET) >= 8:
        ok(f"positive eval set: {len(POSITIVE_EVAL_SET)} questions")
    else:
        fail(f"positive eval set too small: {len(POSITIVE_EVAL_SET)}")

    if len(NEGATIVE_EVAL_SET) >= 1:
        ok(f"negative controls: {len(NEGATIVE_EVAL_SET)} out-of-scope queries")
    else:
        fail("no out-of-scope negative queries in EVAL_SET")

    data_pdfs = sorted(config.DATA_DIR.glob("*.pdf"))
    covered_docs = {q["expected_document"] for q in POSITIVE_EVAL_SET}
    active_docs = {p.name for p in data_pdfs}
    if covered_docs <= active_docs:
        ok(f"eval set documents match data/ corpus ({len(covered_docs)} docs)")
    else:
        fail(f"eval set references docs not in data/: {covered_docs - active_docs}")

    # --- Index sanity ---
    pages = load_pdfs(config.DATA_DIR)
    chunks = chunk_documents(pages)
    ok(f"ingestion produces {len(chunks)} chunks from {len(pages)} pages")

    if not config.VECTOR_DB_DIR.exists():
        fail("vectorstore/ missing — run python ingest.py")
    else:
        ok("vectorstore/ exists")

    vectordb = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=str(config.VECTOR_DB_DIR),
    )
    indexed = vectordb._collection.count()
    if indexed == len(chunks):
        ok(f"indexed chunk count matches pipeline ({indexed})")
    else:
        fail(f"index drift: vectorstore has {indexed} chunks, pipeline produces {len(chunks)}")

    # --- Module 1: Top-K metrics ---
    metrics = {m["k"]: m for m in evaluate_k_values(vectordb)}
    k = config.TOP_K
    if k not in metrics:
        fail(f"TOP_K={k} not evaluated")
    else:
        m = metrics[k]
        ok(
            f"K={k}: Precision@{k}={m['precision']:.1%}, "
            f"PageRecall@{k}={m['page_recall']:.1%}, DocHit@{k}={m['doc_hit']:.1%}"
        )
        if m["doc_hit"] < 0.9:
            fail(f"DocHit@{k} below 90%: {m['doc_hit']:.1%}")
        if m["page_recall"] < 0.7:
            fail(f"PageRecall@{k} below 70%: {m['page_recall']:.1%}")

    # --- Negative controls ---
    neg = evaluate_negative_controls(vectordb)
    failed_neg = [r for r in neg if not r["passed"]]
    if not failed_neg:
        ok(f"all {len(neg)} out-of-scope queries below confidence threshold")
    else:
        for r in failed_neg:
            fail(
                f"out-of-scope query scored too high: top={r['top_score']:.3f} "
                f"for '{r['question'][:50]}...'"
            )

    # --- Module 4: Explainability ---
    sample = retrieve_with_citation(vectordb, POSITIVE_EVAL_SET[0]["question"], k=k)
    if sample and all(
        key in sample[0] for key in ("score", "confidence", "document_name", "page_number", "chunk_id")
    ):
        ok("retrieve_with_citation returns score + confidence + citation metadata")
    else:
        fail("retrieve_with_citation missing required explainability fields")

    # --- Config centralization ---
    if hasattr(config, "TOP_K") and config.TOP_K == k:
        ok(f"config.TOP_K = {config.TOP_K}")
    else:
        fail("config.TOP_K not set or inconsistent")

    print("\n".join(CHECKS))
    if FAILURES:
        print("\n".join(FAILURES), file=sys.stderr)
        print(f"\n{len(FAILURES)} Day 2 check(s) FAILED.", file=sys.stderr)
        return 1
    print(f"\nAll {len(CHECKS)} Day 2 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
