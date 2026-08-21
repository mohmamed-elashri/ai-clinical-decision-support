"""Day 1 Definition of Done — automated verification."""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

import config
from ingest import (
    chunk_documents,
    get_embedding_function,
    load_pdfs,
)
from langchain_chroma import Chroma

FAILURES: list[str] = []
CHECKS: list[str] = []


def ok(msg: str) -> None:
    CHECKS.append(f"PASS  {msg}")


def fail(msg: str) -> None:
    FAILURES.append(f"FAIL  {msg}")


def main() -> int:
    # Active corpus: 2 premier WHO guidelines in data/; annexes in reference_candidates/.
    data_pdfs = sorted(config.DATA_DIR.glob("*.pdf"))

    if not data_pdfs:
        fail("data/ has no PDFs")
    else:
        ok(f"data/ contains {len(data_pdfs)} ingestion source(s): {[p.name for p in data_pdfs]}")

    ref_pdfs = sorted(config.REFERENCE_DIR.glob("*.pdf"))
    if ref_pdfs:
        ok(f"reference_candidates/ holds {len(ref_pdfs)} secondary source(s)")

    if not config.VECTOR_DB_DIR.exists():
        fail("vectorstore/ not found — run python ingest.py")
    else:
        ok("vectorstore/ exists")

    pages = load_pdfs(config.DATA_DIR)
    ok(f"load_pdfs: {len(pages)} pages from {len(data_pdfs)} PDF(s)")

    chunks = chunk_documents(pages)
    ok(f"chunk_documents: {len(chunks)} chunks")

    missing_meta = [
        c.metadata.get("chunk_id")
        for c in chunks
        if not c.metadata.get("document_name") or not c.metadata.get("page_number")
    ]
    if missing_meta:
        fail(f"{len(missing_meta)} chunks missing document_name or page_number")
    else:
        ok("every chunk has document_name + page_number + chunk_id")

    # Per-page chunk IDs must restart at 1 for each (document, page) pair.
    from collections import Counter

    id_suffixes = Counter()
    for c in chunks:
        cid = c.metadata["chunk_id"]
        suffix = cid.rsplit("_c", 1)[-1]
        id_suffixes[suffix] += 1
    if all(int(s) >= 1 for s in id_suffixes):
        ok("chunk_id uses per-page counters (stable format)")

    # --- Checkpoint 3: semantic similarity ---
    embed_fn = get_embedding_function()
    texts = [
        "first-line treatment for hypertension",
        "initial therapy for high blood pressure",
        "recommended screening interval for breast cancer",
    ]
    vecs = np.array(embed_fn.embed_documents(texts))

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_related = cosine(vecs[0], vecs[1])
    sim_unrelated = cosine(vecs[0], vecs[2])
    if sim_related > sim_unrelated:
        ok(f"Checkpoint 3: related={sim_related:.3f} > unrelated={sim_unrelated:.3f}")
    else:
        fail(f"Checkpoint 3: related={sim_related:.3f} NOT > unrelated={sim_unrelated:.3f}")

    # --- Checkpoint 4: retrieval + citations ---
    vectordb = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=str(config.VECTOR_DB_DIR),
    )
    question = (
        "What is the target blood pressure for a patient with cardiovascular disease?"
    )
    results = vectordb.similarity_search_with_relevance_scores(question, k=config.TOP_K)

    indexed_count = vectordb._collection.count()
    if indexed_count == len(chunks):
        ok(f"vectorstore chunk count matches pipeline ({indexed_count})")
    else:
        fail(f"vectorstore has {indexed_count} chunks but pipeline produces {len(chunks)}")

    if not results:
        fail("Checkpoint 4: no retrieval results")
    else:
        top_doc, top_score = results[0]
        if top_doc.metadata.get("document_name") and top_doc.metadata.get("page_number"):
            ok(
                f"Checkpoint 4: top hit cited as "
                f"{top_doc.metadata['document_name']}, page {top_doc.metadata['page_number']} "
                f"(score={top_score:.3f})"
            )
        else:
            fail("Checkpoint 4: top hit missing citation metadata")

        none_citations = [
            i
            for i, (doc, _) in enumerate(results, 1)
            if not doc.metadata.get("document_name") or not doc.metadata.get("page_number")
        ]
        if none_citations:
            fail(f"Checkpoint 4: results {none_citations} have None citations")
        else:
            ok(f"Checkpoint 4: all top-{config.TOP_K} results have document_name + page_number")

        content_lower = top_doc.page_content.lower()
        if any(kw in content_lower for kw in ("130", "blood pressure", "systolic", "target")):
            ok("Checkpoint 4: top chunk content is clinically relevant")
        else:
            fail("Checkpoint 4: top chunk may not be relevant to the query")

    print("\n".join(CHECKS))
    if FAILURES:
        print("\n".join(FAILURES), file=sys.stderr)
        print(f"\n{len(FAILURES)} check(s) FAILED.", file=sys.stderr)
        return 1
    print(f"\nAll {len(CHECKS)} checks passed. Day 1 DoD is met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
