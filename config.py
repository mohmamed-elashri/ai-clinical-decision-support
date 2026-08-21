"""
Central configuration for the AI Clinical Decision Support Lite hackathon
Day 1 ingestion pipeline.

Import this from anywhere in the repo (scripts, notebooks) so paths and
chunking parameters stay consistent across the whole pipeline.
"""

import os
from pathlib import Path


# --- Paths -------------------------------------------------------------
# Repo root = the folder this file lives in.
BASE_DIR = Path(__file__).resolve().parent

# Folder containing the source clinical guideline PDF(s).
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data"))).expanduser()

# Folder where the persisted Chroma vector store is written.
# Production should point at a baked, read-only index and never ingest at startup.
VECTOR_DB_DIR = Path(
    os.environ.get("CHROMA_PATH", str(BASE_DIR / "vectorstore"))
).expanduser()

# Secondary annexes kept out of the active index (documented in design note).
REFERENCE_DIR = BASE_DIR / "reference_candidates"

# Name of the Chroma collection that holds the ingested chunks.
COLLECTION_NAME = "clinical_guidelines"

# --- Chunking ------------------------------------------------------------
# Target chunk size and overlap, expressed in *tokens*. ingest.py converts
# these to an approximate character count for the splitter.
CHUNK_SIZE = 500        # tokens
CHUNK_OVERLAP = 75      # tokens  (~15% overlap — preserves sentence boundaries)
CHARS_PER_TOKEN = 4     # approximation for English clinical text

# --- Embeddings ------------------------------------------------------------
# Local, no-API-key-needed embedding model served via FastEmbed.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Models compared in Day 2 Module 3 (embedding benchmark).
EMBEDDING_BENCHMARK_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
]

# --- Retrieval ----------------------------------------------------------
TOP_K = int(os.environ.get("TOP_K", "3"))
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "dense").strip().lower()
RETRIEVAL_CANDIDATES = 15
EVAL_K_VALUES = (1, 3, 4, 5, 10)
APP_ENV = os.environ.get("APP_ENV", "development").strip().lower()
ALLOW_EXTRACTIVE_FALLBACK = os.environ.get(
    "ALLOW_EXTRACTIVE_FALLBACK",
    "0" if APP_ENV == "production" else "1",
) == "1"

# Minimum relevance score to label a retrieved chunk as "confident".
CONFIDENCE_THRESHOLD = 0.70

# Out-of-scope queries should stay below this top-1 score.
OUT_OF_SCOPE_MAX_SCORE = float(os.environ.get("OUT_OF_SCOPE_MAX_SCORE", "0.65"))

# Dense → re-rank pipeline (fetch wide, return narrow).
RERANK_FETCH_K = 10
RERANK_MODEL_NAME = "ms-marco-TinyBERT-L-2-v2"  # FlashRank ONNX; MiniLM-L-6-v2 zip unavailable


def ensure_dirs() -> None:
    """
    Create required project directories if they do not already exist.

    Call this explicitly only in scripts that write to disk (e.g. ingest.py).
    Avoid running it on every import — directory creation is a side effect
    that should be intentional, not automatic.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
