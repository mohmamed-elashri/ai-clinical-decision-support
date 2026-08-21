"""
chunk_ablation.py — Day 2 Module 2
Ablation study on Chunk Size and Overlap settings.
Creates temporary in-memory collections and evaluates Precision@5 for each config.
To optimize runtime, it runs the ablation on a representative subset of files (or all if fast).
"""
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from evaluation_set import POSITIVE_EVAL_SET
from ingest import clean_page_text, get_embedding_function

# 3 configs to test (sizes in tokens, matched with *4 multiplier to approximate
# characters as in ingest.py).
CONFIGS = [
    {"name": "300_40", "chunk_size": 300, "chunk_overlap": 40},
    {"name": "500_75_current", "chunk_size": 500, "chunk_overlap": 75},
    {"name": "700_100", "chunk_size": 700, "chunk_overlap": 100},
]


def load_ablation_docs(data_dir: Path, limit_files=None) -> list:
    """
    Load PDFs for the ablation study, applying the same text cleaning
    as the main ingest pipeline so results are directly comparable.

    Can optionally limit to the top-N largest files to save time.
    """
    from langchain_community.document_loaders import PyPDFLoader

    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if limit_files:
        pdf_paths = sorted(pdf_paths, key=lambda p: -p.stat().st_size)[:limit_files]
        print(f"  [Ablation Optimization] Using top {limit_files} files for speed.")

    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        for d in pages:
            # Apply the same normalisation as ingest.py so ablation precision
            # numbers are directly comparable to the production index metrics.
            d.page_content = clean_page_text(d.page_content)
            d.metadata["document_name"] = path.name
            d.metadata["page_number"] = d.metadata.get("page", 0) + 1
        docs.extend(pages)
    return docs


def precision_at_k(vectordb, eval_set, k=5) -> float:
    """Calculates Precision@K for a given vector store."""
    # Find which documents are actually in this ablation DB.
    db_metadata = vectordb._collection.get(include=["metadatas"])["metadatas"]
    available_docs = {meta.get("document_name") for meta in db_metadata}

    filtered_eval_set = [
        item for item in eval_set if item["expected_document"] in available_docs
    ]

    if not filtered_eval_set:
        print("  ⚠️ Warning: No evaluation questions match the loaded ablation documents!")
        return 0.0

    precisions = []
    for item in filtered_eval_set:
        results = vectordb.similarity_search_with_relevance_scores(item["question"], k=k)
        relevant = sum(
            1
            for doc, _score in results
            if doc.metadata.get("document_name") == item["expected_document"]
            and doc.metadata.get("page_number") in item["expected_pages"]
        )
        precisions.append(relevant / k)
    return sum(precisions) / len(filtered_eval_set)


if __name__ == "__main__":
    data_dir = Path(config.DATA_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {data_dir}")

    print("Loading documents for ablation study...")
    t0 = time.time()
    # Load all active documents in data/ (2 premier WHO guidelines).
    # Loaded once here — each config only re-splits, not re-loads from disk.
    docs = load_ablation_docs(data_dir)
    print(f"Loaded {len(docs)} pages in {time.time() - t0:.1f}s\n")

    embedding_fn = get_embedding_function()
    results = []

    print("=" * 80)
    print(
        f"{'Config Name':<16} | {'Chunk Size':<10} | {'Overlap':<8} | "
        f"{'Num Chunks':<10} | {'Precision@5':<12} | Time"
    )
    print("-" * 80)

    for cfg in CONFIGS:
        t_start = time.time()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"] * config.CHARS_PER_TOKEN,
            chunk_overlap=cfg["chunk_overlap"] * config.CHARS_PER_TOKEN,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        # Stamp chunk_ids (ablation-specific prefix avoids collision with
        # the production collection).
        for i, chunk in enumerate(chunks):
            doc_name = chunk.metadata.get("document_name", "unknown")
            page_num = chunk.metadata.get("page_number", "?")
            chunk.metadata["chunk_id"] = f"ablation_{cfg['name']}_{doc_name}_p{page_num}_c{i}"

        # Create transient in-memory Chroma collection.
        vdb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_fn,
            collection_name=f"temp_ablation_{cfg['name']}",
            ids=[c.metadata["chunk_id"] for c in chunks],
        )

        score = precision_at_k(vdb, POSITIVE_EVAL_SET, k=5)
        duration = time.time() - t_start

        results.append(
            {
                "name": cfg["name"],
                "chunk_size": cfg["chunk_size"],
                "chunk_overlap": cfg["chunk_overlap"],
                "num_chunks": len(chunks),
                "precision": score,
                "time": duration,
            }
        )

        print(
            f"{cfg['name']:<16} | {cfg['chunk_size']:<10} | {cfg['chunk_overlap']:<8} | "
            f"{len(chunks):<10} | {score:>11.2%} | {duration:.1f}s"
        )

        vdb.delete_collection()

    print("=" * 80)
    best = max(results, key=lambda x: x["precision"])
    print(f"\n🏆 Recommended Config: {best['name']} (size={best['chunk_size']}, overlap={best['chunk_overlap']})")
    print(f"   Precision@5: {best['precision']:.2%}")
    print(f"   Total Chunks: {best['num_chunks']}")
    print("\nDecision Guidance:")
    print(
        "   - If your current config (500_75) is close to or matches the best, "
        "keep it to avoid rebuilding the index."
    )
    print(
        "   - If another config significantly outperforms, "
        "update config.py and rerun ingest.py."
    )
