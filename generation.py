"""Day 3 grounded generation and abstain logic."""

import logging
import os
import re

import config
from dense_reranker import DenseRerankRetriever
from retrieval import retrieve_with_citation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list:
    return re.findall(r"[a-zA-Z]{3,}", text.lower())


def _select_best_sentence(question: str, chunks: list[dict]) -> str:
    """
    Lightweight extractive answer: pick the sentence from the retrieved chunks
    that has the highest token-overlap with the question, weighted slightly by
    sentence length (to favour more complete answers).
    """
    text_blocks = []
    for chunk in chunks:
        text_blocks.extend(
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", chunk.get("text", ""))
            if s.strip()
        )

    if not text_blocks:
        return "The retrieved evidence does not contain a complete answer to this question."

    question_tokens = set(_tokenize(question))
    scored = []
    for sentence in text_blocks:
        sentence_tokens = set(_tokenize(sentence))
        overlap = len(question_tokens & sentence_tokens)
        sentence_len = len(sentence)
        score = overlap * 2 + min(sentence_len, 220) / 80
        scored.append((score, sentence))

    _, best = max(scored, key=lambda item: item[0])
    return best[:500].strip()


def _abstain(reason: str, top_score: float) -> dict:
    """Construct a consistent abstain response."""
    return {
        "status": "abstain",
        "answer": "",
        "evidence": [],
        "citations": [],
        "confidence": "insufficient",
        "reason": reason,
        "top_score": top_score,
        "generation_method": "none",
        "retrieved_chunks": [],
    }


def _build_extractive_response(
    top: dict, top_score: float, used_chunks: list, question: str
) -> dict:
    """Build an extractive fallback response from retrieved chunks."""
    answer_text = _select_best_sentence(question, used_chunks)
    seen: set = set()
    citations = []
    evidence_list = []

    for chunk in used_chunks:
        key = (chunk.get("document_name"), chunk.get("page_number"))
        if key not in seen:
            seen.add(key)
            citations.append(
                {
                    "document_name": chunk.get("document_name"),
                    "page_number": chunk.get("page_number"),
                    "chunk_id": chunk.get("chunk_id"),
                }
            )
        if txt := chunk.get("text", ""):
            evidence_list.append(txt[:300].strip())

    return {
        "status": "grounded",
        "answer": answer_text,
        "evidence": evidence_list,
        "citations": citations,
        "confidence": top.get("confidence", "confident"),
        "top_score": top_score,
        "generation_method": "extractive_fallback",
        "retrieved_chunks": used_chunks,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_question(
    vectordb,
    question: str,
    k: int | None = None,
    use_reranker: bool = False,
    reranker=None,
) -> dict:
    """
    Return a grounded answer with citation metadata, or abstain if evidence
    is weak or the query is out of scope.

    Flow:
        1. Retrieve top-K chunks from the vector store.
        2. Abstain immediately if no chunks are returned.
        3. Abstain if the top relevance score is below OUT_OF_SCOPE_MAX_SCORE.
        4. Try live LLM generation via NIM (if NV_API_KEY is configured).
          5. Abstain when live generation is unavailable unless an explicit
              non-production extractive fallback is enabled.
    """
    retrieved = retrieve_with_citation(
        vectordb, question, k=k, use_reranker=use_reranker, reranker=reranker
    )
    if not retrieved:
        return _abstain("No relevant guideline evidence retrieved.", top_score=0.0)

    top = retrieved[0]
    # Dense relevance is the calibrated scope/abstention signal. A reranker
    # score has a different scale and is used only to order candidates.
    top_score = float(top.get("dense_score", top.get("score", 0.0)))
    if top_score < config.OUT_OF_SCOPE_MAX_SCORE:
        return _abstain(
            f"Top retrieval score ({top_score:.3f}) below threshold "
            f"({config.OUT_OF_SCOPE_MAX_SCORE:.2f}).",
            top_score=top_score,
        )

    used_chunks = retrieved[: min(3, len(retrieved))]

    # Try live LLM generation if the canonical or legacy key is configured.
    if os.environ.get("NVIDIA_API_KEY") or os.environ.get("NV_API_KEY"):
        try:
            from llm_client import build_grounded_prompt, call_nim_chat, parse_and_clean_json
            from llm_client import verify_response_provenance

            prompt_text = build_grounded_prompt(question, used_chunks)
            raw = call_nim_chat(
                prompt_text,
                timeout=int(os.environ.get("LLM_TIMEOUT", "3")),
            )
            res = parse_and_clean_json(raw, top_score=top_score)
            provenance_errors = verify_response_provenance(res, used_chunks)
            if provenance_errors:
                raise ValueError(
                    "LLM response failed evidence provenance: "
                    + "; ".join(provenance_errors)
                )
            res["generation_method"] = "llm"
            res["retrieved_chunks"] = used_chunks
            return res
        except Exception as exc:
            # Log the failure so operators can see it — do NOT swallow silently.
            logger.warning(
                "LLM generation failed (%s). Falling back to extractive answer.", exc
            )

    # Extractive fallback is opt-in for production because it can omit
    # qualifiers that are clinically important.
    if not config.ALLOW_EXTRACTIVE_FALLBACK:
        return _abstain(
            "Live LLM generation is unavailable and extractive fallback is disabled.",
            top_score=top_score,
        )

    return _build_extractive_response(top, top_score, used_chunks, question)


def answer_question_cli(question: str, k: int | None = None, use_reranker: bool = False):
    """Convenience wrapper for the project CLI."""
    from query import load_index

    vectordb = load_index()
    return answer_question(vectordb, question, k=k, use_reranker=use_reranker)


if __name__ == "__main__":
    test_question = "What is the target blood pressure for a patient with cardiovascular disease?"
    result = answer_question_cli(test_question)
    print(result["status"].upper())
    print(result["answer"])
    print("Citations:")
    for c in result["citations"]:
        print(f"  - {c['document_name']} (p.{c['page_number']})")
