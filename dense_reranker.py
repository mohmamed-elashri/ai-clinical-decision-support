"""
dense_reranker.py — Dense-first retrieval with cross-encoder re-ranking.

Fetches top-N dense candidates only (no BM25 mixing), then re-ranks with a
lightweight cross-encoder. Avoids the score-scale mismatch that hurt the
hybrid+reranker benchmark (716ms, low precision).
"""

from flashrank import Ranker, RerankRequest

import config
from ingest import get_embedding_function
from langchain_chroma import Chroma


class DenseRerankRetriever:
    """Dense vector search → cross-encoder re-rank → top-K."""

    def __init__(
        self,
        vectordb=None,
        rerank_model_name: str | None = None,
        fetch_k: int | None = None,
    ):
        self.fetch_k = fetch_k or config.RERANK_FETCH_K
        model = rerank_model_name or config.RERANK_MODEL_NAME

        if vectordb is None:
            self.vectordb = Chroma(
                collection_name=config.COLLECTION_NAME,
                embedding_function=get_embedding_function(),
                persist_directory=str(config.VECTOR_DB_DIR),
            )
        else:
            self.vectordb = vectordb

        self.ranker = Ranker(model_name=model)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or config.TOP_K
        dense_hits = self.vectordb.similarity_search_with_relevance_scores(
            query, k=self.fetch_k
        )
        if not dense_hits:
            return []

        passages = []
        for doc, dense_score in dense_hits:
            meta = {
                "document_name": doc.metadata.get("document_name"),
                "page_number": doc.metadata.get("page_number"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "dense_score": round(float(dense_score), 3),
            }
            passages.append({"id": meta["chunk_id"], "text": doc.page_content, "meta": meta})

        reranked = self.ranker.rerank(RerankRequest(query=query, passages=passages))

        results = []
        for rank, item in enumerate(reranked[:top_k], start=1):
            rerank_score = round(float(item["score"]), 3)
            meta = item["meta"]
            # Re-ranker scores are not cosine similarities — use dense_score for confidence.
            dense_score = meta.get("dense_score", 0.0)
            results.append(
                {
                    "rank": rank,
                    "text": item["text"],
                    "document_name": meta.get("document_name"),
                    "page_number": meta.get("page_number"),
                    "chunk_id": meta.get("chunk_id"),
                    "score": rerank_score,
                    "dense_score": dense_score,
                    "confidence": (
                        "confident"
                        if dense_score >= config.CONFIDENCE_THRESHOLD
                        else "uncertain"
                    ),
                    "retrieval_method": f"Dense + Re-ranker ({config.RERANK_MODEL_NAME})",
                }
            )
        return results


if __name__ == "__main__":
    retriever = DenseRerankRetriever()
    q = "What BP target applies to patients with cardiovascular disease?"
    print(f"Query: {q}\n")
    for r in retriever.retrieve(q):
        print(
            f"[{r['rank']}] rerank={r['score']:.3f} dense={r['dense_score']:.3f} "
            f"| {r['document_name']} p.{r['page_number']}"
        )
