"""
retrieval.py — Day 2 Module 4
Clinical retrieval wrapper with relevance score, confidence threshold, citation metadata, and visual layout.
"""
import config

CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD
_HYBRID_CACHE = {}


def retrieve_with_citation(
    vectordb,
    question: str,
    k: int | None = None,
    use_reranker: bool = False,
    reranker=None,
    retrieval_mode: str | None = None,
):
    """
    Retrieves top K clinical chunks with citation metadata, relevance scores,
    and confidence labels. Supports the optional dense-first reranker for hard
    paraphrase queries.
    """
    k = config.TOP_K if k is None else k
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    mode = (retrieval_mode or config.RETRIEVAL_MODE).strip().lower()
    if mode not in {"dense", "rrf"}:
        raise ValueError("retrieval_mode must be 'dense' or 'rrf'")
    if mode == "rrf":
        from hybrid_retriever import HybridClinicalRetriever

        cache_key = (id(vectordb), use_reranker)
        hybrid = _HYBRID_CACHE.get(cache_key)
        if hybrid is None:
            hybrid = HybridClinicalRetriever(
                vectordb=vectordb,
                load_reranker=use_reranker,
            )
            _HYBRID_CACHE[cache_key] = hybrid
        return hybrid.retrieve(
            question,
            top_k=k,
            fetch_candidates=config.RETRIEVAL_CANDIDATES,
            use_reranker=use_reranker,
        )
    if use_reranker:
        if reranker is None:
            from dense_reranker import DenseRerankRetriever
            reranker = DenseRerankRetriever(vectordb=vectordb)
        results = reranker.retrieve(question, top_k=k)
        return results

    # similarity_search_with_relevance_scores returns a list of (Document, score) tuples
    dense_results = vectordb.similarity_search_with_relevance_scores(question, k=k)
    output = []
    for rank, (doc, score) in enumerate(dense_results, start=1):
        output.append({
            "rank": rank,
            "text": doc.page_content,
            "document_name": doc.metadata.get("document_name", "unknown"),
            "page_number": doc.metadata.get("page_number", "?"),
            "section": doc.metadata.get("section", None),  # Can be expanded during layout extraction
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            "score": round(score, 3),
            "dense_score": round(score, 3),
            "confidence": "confident" if score >= CONFIDENCE_THRESHOLD else "uncertain",
        })
    return output

def print_retrieval_view(question: str, results: list):
    """
    Module 4 requirement: Visual printout of retrieved chunks with confidence, score, and citation.
    Runs before LLM generation to ensure explainability.
    """
    print("\n" + "=" * 80)
    print(f" CLINICAL QUERY: {question}")
    print("=" * 80)
    for r in results:
        loc = f"{r['document_name']} (p. {r['page_number']})"
        score_info = f"Score: {r['score']:.3f}"
        conf_marker = "⭐ [CONFIDENT]" if r["confidence"] == "confident" else "❓ [UNCERTAIN]"
        
        print(f"\n[{r['rank']}] {conf_marker} | {score_info} | Source: {loc}")
        print(f"    Chunk ID: {r['chunk_id']}")
        print("-" * 80)
        # Show first 300 characters of the text for the preview
        preview_text = r['text'].replace('\n', ' ').strip()
        if len(preview_text) > 280:
            preview_text = preview_text[:277] + "..."
        print(f"    Content: \"{preview_text}\"")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    from query import load_index

    print("Initializing Chroma Vector Database...")
    vectordb = load_index()

    test_question = "What is the recommended target blood pressure for patients with cardiovascular disease?"
    print("Running retrieval test...")
    results = retrieve_with_citation(vectordb, test_question)
    print_retrieval_view(test_question, results)
