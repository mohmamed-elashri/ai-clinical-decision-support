"""
Day 1 ingestion pipeline: PDF -> normalized pages -> citation-ready chunks
-> embeddings -> persisted Chroma vector index.

Run directly:
    python ingest.py
to (re)build the vector index from every PDF in config.DATA_DIR.

Day1_Task1_Document_Ingestion.ipynb imports every public function below
directly, so keep the function names and the metadata keys
('document_name', 'page_number', 'chunk_id') stable.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import List

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


# Repeated PDF boilerplate stripped before chunking (improves embedding quality).
_HEADER_PATTERNS = [
    re.compile(r"GUIDELINE FOR THE PHARMACOLOGICAL TREATMENT OF HYPERTENSION IN ADULTS", re.I),
    re.compile(r"WHO HEARTS TECHNICAL PACKAGE", re.I),
    re.compile(r"HEARTS TECHNICAL PACKAGE", re.I),
]
_PAGE_NUM_ONLY = re.compile(r"^\s*\d+\s*$", re.M)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def clean_page_text(text: str) -> str:
    """Remove repeated headers, orphan page numbers, and excess whitespace."""
    cleaned = text
    for pattern in _HEADER_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = _PAGE_NUM_ONLY.sub("", cleaned)
    cleaned = _MULTI_SPACE.sub(" ", cleaned)
    return cleaned.strip()


def load_pdfs(data_dir: Path) -> List[Document]:
    """
    Load every PDF in `data_dir` and return one Document per page, with
    citation-ready metadata stamped on:
      - document_name: the PDF's filename (e.g. 'WHO_Hypertension_Guideline_2021.pdf')
      - page_number:   1-indexed page number (PyPDFLoader's own 'page' key
                        is 0-indexed, which is not human/citation friendly)
    """
    data_dir = Path(data_dir)
    pdf_paths = sorted(data_dir.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {data_dir}. Add at least one guideline PDF there."
        )

    all_pages: List[Document] = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.page_content = clean_page_text(page.page_content)
            page.metadata["document_name"] = pdf_path.name
            # PyPDFLoader's 'page' key is 0-indexed; citations should read
            # naturally ("page 3"), so we store a separate 1-indexed field.
            page.metadata["page_number"] = page.metadata.get("page", 0) + 1
        all_pages.extend(pages)

    return all_pages


def chunk_documents(pages: List[Document]) -> List[Document]:
    """
    Split pages into section-aware chunks and attach a stable chunk_id to
    each one. Metadata already on each page (document_name, page_number)
    is preserved automatically by split_documents().
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE * config.CHARS_PER_TOKEN,
        chunk_overlap=config.CHUNK_OVERLAP * config.CHARS_PER_TOKEN,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    page_counters: defaultdict[tuple[str, object], int] = defaultdict(int)
    for chunk in chunks:
        doc_name = chunk.metadata.get("document_name", "unknown")
        page_num = chunk.metadata.get("page_number", "?")
        key = (doc_name, page_num)
        page_counters[key] += 1
        chunk.metadata["chunk_id"] = f"{doc_name}_p{page_num}_c{page_counters[key]}"

    return chunks


def get_embedding_function() -> FastEmbedEmbeddings:
    """
    Local embedding model — no API key required. First call downloads and
    caches the model (~100 MB) under ~/.cache/fastembed/.
    """
    return FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


def build_index(chunks: List[Document]) -> Chroma:
    """
    Embed every chunk and persist it into a local Chroma collection.
    Clears old collection first to prevent stale chunk accumulation.
    """
    embedding_fn = get_embedding_function()

    client = chromadb.PersistentClient(path=str(config.VECTOR_DB_DIR))
    try:
        client.delete_collection(name=config.COLLECTION_NAME)
        print(f"  -> Cleared existing collection '{config.COLLECTION_NAME}'")
    except Exception as exc:
        # Collection does not exist yet — expected on the first run.
        print(f"  -> No existing collection to clear ({exc})")

    vectordb = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
        client=client,
    )
    # Add texts in batches of 32 to avoid ONNX memory spike.
    batch_size = 32
    total_batches = (len(chunks) - 1) // batch_size + 1
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  -> Indexing batch {batch_num}/{total_batches} ({len(batch)} chunks)...", flush=True)
        vectordb.add_documents(
            documents=batch,
            ids=[c.metadata["chunk_id"] for c in batch],
        )
    return vectordb


def main():
    config.ensure_dirs()
    print(f"Loading PDFs from: {config.DATA_DIR}")
    pages = load_pdfs(config.DATA_DIR)
    print(f"  -> {len(pages)} pages loaded")

    print("Chunking...")
    chunks = chunk_documents(pages)
    print(f"  -> {len(chunks)} chunks created")

    print("Embedding + building vector index (first run downloads the model)...")
    vectordb = build_index(chunks)
    print(f"  -> Index persisted to: {config.VECTOR_DB_DIR}")

    return vectordb


if __name__ == "__main__":
    main()
