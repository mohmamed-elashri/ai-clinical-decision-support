"""
hybrid_retriever.py — Advanced Clinical Retrieval Pipeline
Combines:
1. Dense Vector Search (ChromaDB + FastEmbed bge-small)
2. Sparse Keyword Search (BM25 for exact drug names & clinical thresholds)
3. Reciprocal Rank Fusion (RRF)
4. FastRank Re-ranker (ms-marco-MiniLM-L-6-v2 ONNX)
"""

from typing import Any, Dict, List, Tuple

from flashrank import Ranker, RerankRequest
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

import config
from ingest import get_embedding_function


_CLINICAL_QUERY_TERMS = {
    "cvd": "cardiovascular disease heart disease",
    "heart disease": "cardiovascular disease CVD",
    "blood pressure target": "BP treatment goal systolic target",
    "target blood pressure": "BP treatment goal systolic target",
    "first-line": "initial pharmacological treatment antihypertensive classes",
    "first line": "initial pharmacological treatment antihypertensive classes",
    "lifestyle": "non-pharmacological behavioral intervention",
    "combination therapy": "two-drug combination treatment intensification",
    "follow-up": "reassessment repeat measurement visit",
    "hearts": "WHO primary health care hypertension protocol",
    "titration": "medication escalation treatment step algorithm",
    "risk assessment": "cardiovascular risk evaluation CVD risk",
}


def expand_clinical_query(query: str) -> str:
    """Add retrieval synonyms without changing the original clinical question."""
    lowered = query.lower()
    additions = [terms for trigger, terms in _CLINICAL_QUERY_TERMS.items() if trigger in lowered]
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


class HybridClinicalRetriever:
    def __init__(
        self,
        vectordb=None,
        rerank_model_name: str = "ms-marco-TinyBERT-L-2-v2",
        load_reranker: bool = True,
    ):
        """Initializes the vector store, fetches all chunks for BM25, and loads FlashRank."""
        if vectordb is None:
            self.vectordb = Chroma(
                collection_name=config.COLLECTION_NAME,
                embedding_function=get_embedding_function(),
                persist_directory=str(config.VECTOR_DB_DIR),
            )
        else:
            self.vectordb = vectordb

        # Fetch all chunks & metadata from Chroma collection for BM25.
        # NOTE: _collection is a semi-internal API; if ChromaDB breaks it in a
        # future release, replace with a public langchain_chroma equivalent.
        print("  [HybridRetriever] Loading all chunks from VectorDB for BM25 index...")
        db_data = self.vectordb._collection.get(include=["documents", "metadatas"])
        self.documents: List[str] = db_data["documents"]
        self.metadatas: List[Dict[str, Any]] = db_data["metadatas"]
        self.chunk_ids: List[str] = db_data["ids"]

        tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"  [HybridRetriever] BM25 Index built over {len(self.documents)} chunks.")

        # Initialize FlashRank Reranker.
        self.ranker = None
        if load_reranker:
            print(f"  [HybridRetriever] Loading FlashRank Reranker ({rerank_model_name})...")
            self.ranker = Ranker(model_name=rerank_model_name)
        print("  [HybridRetriever] Initialization Complete ✅")

    def _bm25_search(
        self, query: str, top_n: int = 10
    ) -> List[Tuple[Dict[str, Any], str, float]]:
        """Performs BM25 keyword search."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero matches.
                results.append(
                    (
                        {
                            "document_name": self.metadatas[idx].get("document_name"),
                            "page_number": self.metadatas[idx].get("page_number"),
                            "chunk_id": self.metadatas[idx].get("chunk_id"),
                        },
                        self.documents[idx],
                        float(scores[idx]),
                    )
                )
        return results

    def _dense_search(
        self, query: str, top_n: int = 10
    ) -> List[Tuple[Dict[str, Any], str, float]]:
        """Performs Dense Vector similarity search."""
        results = self.vectordb.similarity_search_with_relevance_scores(query, k=top_n)
        formatted = []
        for doc, score in results:
            meta = {
                "document_name": doc.metadata.get("document_name"),
                "page_number": doc.metadata.get("page_number"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "dense_score": round(float(score), 3),
            }
            formatted.append((meta, doc.page_content, float(score)))
        return formatted

    def rrf_merge(
        self,
        dense_results: list,
        bm25_results: list,
        k_rrf: int = 60,
        top_n: int = 10,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list:
        """Combines Dense and BM25 search results using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = {}
        items_map: Dict[str, Tuple] = {}

        for rank, (meta, text, _) in enumerate(dense_results, start=1):
            chunk_id = meta["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (
                dense_weight / (k_rrf + rank)
            )
            items_map[chunk_id] = (meta, text)

        for rank, (meta, text, _) in enumerate(bm25_results, start=1):
            chunk_id = meta["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (
                sparse_weight / (k_rrf + rank)
            )
            if chunk_id in items_map:
                existing_meta, existing_text = items_map[chunk_id]
                merged_meta = dict(existing_meta)
                merged_meta.update({key: value for key, value in meta.items() if value is not None})
                items_map[chunk_id] = (merged_meta, existing_text)
            else:
                items_map[chunk_id] = (meta, text)

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        merged = []
        for cid, rrf_score in sorted_chunks:
            meta, text = items_map[cid]
            meta = dict(meta)
            meta["rrf_score"] = round(float(rrf_score), 6)
            merged.append((meta, text, rrf_score))
        return merged

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        fetch_candidates: int = 15,
        use_reranker: bool = True,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        expand_query: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Executes Hybrid Retrieval (Vector + BM25) + FlashRank Re-ranking.
        Returns top_k clinical chunks formatted for explainability & downstream LLM.
        """
        retrieval_query = expand_clinical_query(query) if expand_query else query

        # 1. Fetch dense vector candidates.
        dense_hits = self._dense_search(retrieval_query, top_n=fetch_candidates)

        # 2. Fetch sparse BM25 candidates.
        bm25_hits = self._bm25_search(retrieval_query, top_n=fetch_candidates)

        # 3. Merge via RRF.
        candidates = self.rrf_merge(
            dense_hits,
            bm25_hits,
            top_n=fetch_candidates,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

        if not candidates:
            return []

        # 4. Re-rank using FlashRank Cross-Encoder.
        if use_reranker:
            if self.ranker is None:
                raise RuntimeError("Hybrid reranker was not loaded; initialize with load_reranker=True.")
            passages = [
                {"id": meta["chunk_id"], "text": text, "meta": meta}
                for meta, text, _ in candidates
            ]
            rerank_request = RerankRequest(query=query, passages=passages)
            reranked_output = self.ranker.rerank(rerank_request)

            final_results = []
            for rank, item in enumerate(reranked_output[:top_k], start=1):
                rerank_score = round(float(item["score"]), 3)
                meta = item["meta"]
                dense_score = meta.get("dense_score", 0.0)
                final_results.append(
                    {
                        "rank": rank,
                        "text": item["text"],
                        "document_name": meta.get("document_name"),
                        "page_number": meta.get("page_number"),
                        "chunk_id": meta.get("chunk_id"),
                        "score": rerank_score,
                        "rerank_score": rerank_score,
                        "dense_score": dense_score,
                        "rrf_score": meta.get("rrf_score", 0.0),
                        # Use config threshold — consistent with retrieval.py and dense_reranker.py.
                        "confidence": (
                            "confident"
                            if dense_score >= config.CONFIDENCE_THRESHOLD
                            else "uncertain"
                        ),
                        "retrieval_method": "Hybrid + Re-ranker (RRF + FlashRank)",
                    }
                )
            return final_results
        else:
            # RRF order without Re-ranker.
            final_results = []
            for rank, (meta, text, rrf_score) in enumerate(candidates[:top_k], start=1):
                score = round(float(rrf_score), 3)
                final_results.append(
                    {
                        "rank": rank,
                        "text": text,
                        "document_name": meta.get("document_name"),
                        "page_number": meta.get("page_number"),
                        "chunk_id": meta.get("chunk_id"),
                        "score": score,
                        "rrf_score": score,
                        "dense_score": meta.get("dense_score", 0.0),
                        "confidence": (
                            "confident"
                            if meta.get("dense_score", 0.0) >= config.CONFIDENCE_THRESHOLD
                            else "uncertain"
                        ),
                        "retrieval_method": "Hybrid (RRF)",
                    }
                )
            return final_results


if __name__ == "__main__":
    retriever = HybridClinicalRetriever()
    test_q = "What is the recommended target blood pressure for patients with cardiovascular disease?"
    print(f"\nTesting Hybrid Retriever for Query: '{test_q}'\n")
    results = retriever.retrieve(test_q, top_k=3)

    for r in results:
        print(f"[{r['rank']}] Score: {r['score']} | {r['confidence']} | {r['document_name']} (p. {r['page_number']})")
        print(f"    Content snippet: {r['text'][:150]}...\n")
