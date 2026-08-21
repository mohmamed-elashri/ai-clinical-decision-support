"""
query.py — Query and retrieval interface for Day 2 Notebook and pipeline.
"""
import config
from generation import answer_question
from ingest import get_embedding_function
from langchain_chroma import Chroma


def load_index() -> Chroma:
    """
    Loads the persisted Chroma vector database index.
    """
    embedding_fn = get_embedding_function()
    vectordb = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=str(config.VECTOR_DB_DIR),
    )
    return vectordb


def retrieve(vectordb, question: str, k: int | None = None):
    """
    Retrieves the top-k most relevant document chunks along with their relevance scores.
    Returns a list of (Document, score) tuples.
    """
    k = k or config.TOP_K
    return vectordb.similarity_search_with_relevance_scores(question, k=k)


def answer(question: str, k: int | None = None, use_reranker: bool = False):
    """Grounded answer flow for the Day 3 clinical RAG pipeline."""
    vectordb = load_index()
    return answer_question(vectordb, question, k=k, use_reranker=use_reranker)


if __name__ == "__main__":
    vdb = load_index()
    q = "What is the target blood pressure for a patient with cardiovascular disease?"
    print(f"Query: {q}")
    results = retrieve(vdb, q)
    for doc, score in results:
        p = doc.metadata.get("page_number", "?")
        d = doc.metadata.get("document_name", "?")
        print(f"  Score: {score:.3f} | {d} (p.{p})")

    print("\nGrounded answer:")
    response = answer_question(vdb, q)
    print(response["status"].upper())
    print(response["answer"])
    for citation in response["citations"]:
        print(f"  - {citation['document_name']} (p.{citation['page_number']})")
