"""
app.py — Flask REST API backend for AI Clinical Decision Support Lite.

Wraps the existing RAG pipeline (query, retrieval, generation) behind a
small HTTP API so that the Gradio frontend (or any other client) can call it.

Environment variables (all loaded from .env via python-dotenv):
    NVIDIA_API_KEY      — NVIDIA NIM API key (required; NV_API_KEY is legacy).
    NV_LLM_ENDPOINT     — Chat completions endpoint (optional, has default).
    NVIDIA_MODEL        — Model slug (optional, NV_MODEL is legacy).
    APP_DEBUG           — Set to "1" to enable Flask debug mode (default: off).
    FLASK_PORT          — Port to listen on (default: 5000; APP_PORT is legacy).
    CORS_ORIGINS        — Comma-separated allowed origins (default: all origins "*").
    LLM_TIMEOUT         — NIM API call timeout in seconds (default: 30).
"""

import json
import hashlib
import hmac
import logging
import os
import time
import uuid
from pathlib import Path
from collections import defaultdict, deque
from threading import Lock

# Load .env before any other project imports so env vars are available.
from dotenv import load_dotenv

load_dotenv()

import jsonschema
from flask import Flask, jsonify, request
from flask_cors import CORS

# Project modules -- import after load_dotenv so env vars propagate.
from generation import answer_question
from query import load_index
from retrieval import retrieve_with_citation

# -- Logging setup -------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("clinical_api")

# -- Schema setup --------------------------------------------------------------
_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "response_schema.json"
with open(_SCHEMA_PATH, "r", encoding="utf-8") as _f:
    _RESPONSE_SCHEMA = json.load(_f)
_VALIDATOR = jsonschema.Draft7Validator(_RESPONSE_SCHEMA)

# -- Vector index -- loaded lazily --------------------------------------------
_vectordb = None
_chunk_count: int | None = None
_index_lock = Lock()


def _collection_count(vectordb) -> int:
    """Return collection count behind one compatibility boundary."""
    collection = getattr(vectordb, "_collection", None)
    if collection is None or not hasattr(collection, "count"):
        logger.warning("Vector store does not expose a collection count API.")
        return -1
    return int(collection.count())


def get_vectordb():
    """Load the vector index once, on first use, instead of at module import."""
    global _vectordb, _chunk_count
    if _vectordb is None:
        with _index_lock:
            if _vectordb is None:
                logger.info("Loading Chroma vector index ...")
                t0 = time.time()
                _vectordb = load_index()
                _chunk_count = _collection_count(_vectordb)
                logger.info(
                    "Vector index ready: %d chunks (%.1fs)",
                    _chunk_count,
                    time.time() - t0,
                )
    return _vectordb

# -- Flask app -----------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_REQUEST_BYTES", str(32 * 1024))
)

# Configure CORS. Default to local development origins; production must be explicit.
_cors_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:7860,http://127.0.0.1:7860",
)
if os.environ.get("APP_ENV", "").lower() == "production" and _cors_origins == "*":
    raise RuntimeError("CORS_ORIGINS='*' is not allowed when APP_ENV=production.")
_origins = [o.strip() for o in _cors_origins.split(",")] if _cors_origins != "*" else "*"
CORS(app, resources={r"/api/*": {"origins": _origins}})

# -- Constants -----------------------------------------------------------------
_MAX_QUESTION_LEN = 1000
_DEFAULT_K = 3
_MAX_K = 20
_API_AUTH_TOKEN = os.environ.get("API_AUTH_TOKEN", "")
_RATE_LIMIT_PER_MINUTE = int(os.environ.get("API_RATE_LIMIT_PER_MINUTE", "60"))
_REDIS_URL = os.environ.get("REDIS_URL", "")
_redis_client = None
_request_times: dict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = Lock()

if os.environ.get("APP_ENV", "development").lower() == "production":
    if os.environ.get("APP_DEBUG", "0") == "1":
        raise RuntimeError("APP_DEBUG=1 is not allowed in production.")
    if len(_API_AUTH_TOKEN) < 32:
        raise RuntimeError("API_AUTH_TOKEN must be at least 32 characters in production.")
    if not os.environ.get("CORS_ORIGINS"):
        raise RuntimeError("CORS_ORIGINS must be explicitly configured in production.")
    if os.environ.get("ALLOW_EXTRACTIVE_FALLBACK", "0") == "1":
        raise RuntimeError("ALLOW_EXTRACTIVE_FALLBACK=1 is not allowed in production.")
    if os.environ.get("REQUIRE_REDIS", "0") == "1" and not _REDIS_URL:
        raise RuntimeError("REDIS_URL is required when REQUIRE_REDIS=1.")


def _rate_limit(client_key: str) -> str:
    """Return allowed, limited, or unavailable for the current client window."""
    global _redis_client
    if _REDIS_URL:
        try:
            if _redis_client is None:
                import redis

                _redis_client = redis.Redis.from_url(_REDIS_URL, decode_responses=True)
            digest = hashlib.sha256(client_key.encode("utf-8")).hexdigest()[:24]
            bucket = int(time.time() // 60)
            key = f"clinical-api:rate:{digest}:{bucket}"
            count = _redis_client.incr(key)
            if count == 1:
                _redis_client.expire(key, 61)
            return "limited" if count > _RATE_LIMIT_PER_MINUTE else "allowed"
        except Exception as exc:
            logger.error("Rate-limit backend unavailable: %s", exc)
            return "unavailable"

    now = time.monotonic()
    with _rate_limit_lock:
        timestamps = _request_times[client_key]
        while timestamps and now - timestamps[0] >= 60:
            timestamps.popleft()
        if len(timestamps) >= _RATE_LIMIT_PER_MINUTE:
            return "limited"
        timestamps.append(now)
    return "allowed"


@app.before_request
def protect_api():
    if not request.path.startswith("/api/"):
        return None

    supplied_request_id = request.headers.get("X-Request-ID", "")
    try:
        request.request_id = str(uuid.UUID(supplied_request_id))
    except ValueError:
        request.request_id = str(uuid.uuid4())
    if request.path not in {"/api/live", "/api/health"}:
        if os.environ.get("APP_ENV", "development").lower() == "production" and not _API_AUTH_TOKEN:
            logger.error("API_AUTH_TOKEN is required in production request_id=%s", request.request_id)
            return jsonify({"error": "API authentication is not configured."}), 503
        if _API_AUTH_TOKEN and not hmac.compare_digest(
            request.headers.get("X-API-Key", ""), _API_AUTH_TOKEN
        ):
            return jsonify({"error": "Authentication required."}), 401

    client_key = request.remote_addr or "unknown"
    rate_status = _rate_limit(client_key)
    if rate_status == "unavailable":
        return jsonify({"error": "Rate-limit service unavailable."}), 503
    if rate_status == "limited":
        return jsonify({"error": "Rate limit exceeded."}), 429
    return None


@app.after_request
def add_request_id(response):
    request._audit_status = response.status_code
    request_id = getattr(request, "request_id", None)
    if request_id:
        response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    if os.environ.get("APP_ENV", "development").lower() == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.teardown_request
def audit_request(error):
    """Emit an audit event without recording question text or response content."""
    if not request.path.startswith("/api/"):
        return
    request_id = getattr(request, "request_id", "unknown")
    status_code = getattr(request, "_audit_status", 500 if error else 200)
    logger.info(
        "AUDIT request_id=%s method=%s path=%s status=%s remote=%s error=%s",
        request_id,
        request.method,
        request.path,
        status_code,
        request.remote_addr or "unknown",
        type(error).__name__ if error else "none",
    )


def _validate_question(q: str):
    """Return an error message string if the question is invalid, else None."""
    if not q or not q.strip():
        return "Question must not be empty."
    if len(q) > _MAX_QUESTION_LEN:
        return (
            f"Question exceeds the maximum allowed length of {_MAX_QUESTION_LEN} characters."
        )
    return None


def _parse_request_options(body):
    """Validate and normalize common request fields for retrieval endpoints."""
    if not isinstance(body, dict):
        return None, "Request body must be a JSON object."

    question = body.get("question", "")
    if not isinstance(question, str):
        return None, "Question must be a string."

    question_error = _validate_question(question)
    if question_error:
        return None, question_error

    raw_k = body.get("k", _DEFAULT_K)
    if isinstance(raw_k, bool) or not isinstance(raw_k, int):
        return None, "k must be an integer between 1 and 20."
    if not 1 <= raw_k <= _MAX_K:
        return None, f"k must be between 1 and {_MAX_K}."

    use_reranker = body.get("use_reranker", False)
    if not isinstance(use_reranker, bool):
        return None, "use_reranker must be a JSON boolean."

    return (question, raw_k, use_reranker), None


def _validate_response(result: dict):
    """Return a list of schema violation messages (empty if valid)."""
    return [e.message for e in _VALIDATOR.iter_errors(result)]


# -- Endpoints -----------------------------------------------------------------

@app.get("/api/live")
def live():
    """Cheap liveness probe that never initializes external dependencies."""
    return jsonify({"status": "ok"}), 200


@app.get("/api/health")
def health():
    """Liveness + readiness probe."""
    try:
        vectordb = get_vectordb()
        chunks_indexed = _chunk_count if _chunk_count is not None else _collection_count(vectordb)
        ready = chunks_indexed > 0
    except Exception as exc:
        logger.error("Health check failed while loading vector index: %s", exc)
        return jsonify({"status": "error", "ready": False}), 503

    return jsonify(
        {
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "chunks_indexed": chunks_indexed,
            "llm_available": bool(
                os.environ.get("NVIDIA_API_KEY") or os.environ.get("NV_API_KEY")
            ),
            "retrieval_mode": os.environ.get("RETRIEVAL_MODE", "dense"),
        }
    ), 200 if ready else 503


@app.post("/api/ask")
def ask():
    """
    Body (JSON):
        question     str   required
        k            int   optional (default 3)
        use_reranker bool  optional (default false)

    Returns the grounded answer dict (validated against response_schema.json)
    extended with "generation_method": "llm" | "extractive_fallback" | "none".
    """
    body = request.get_json(silent=True)
    options, err = _parse_request_options(body)
    if err:
        request._audit_status = 400
        return jsonify({"error": err}), 400
    question, k, use_reranker = options

    request_id = getattr(request, "request_id", "unknown")
    t_start = time.perf_counter()
    logger.info("ASK request_id=%s question_length=%d k=%d reranker=%s", request_id, len(question), k, use_reranker)

    try:
        vectordb = get_vectordb()
        result: dict = answer_question(
            vectordb,
            question,
            k=k,
            use_reranker=use_reranker,
        )
    except Exception as exc:
        logger.error("answer_question raised unexpectedly: %s", exc)
        request._audit_status = 500
        return jsonify({"error": "Internal pipeline error. Please try again."}), 500

    latency_ms = (time.perf_counter() - t_start) * 1000

    generation_method = result.get("generation_method", "none")

    # Schema validation -- never pass through invalid payloads
    violations = _validate_response(result)
    if violations:
        logger.error("Schema violation request_id=%s: %s", request_id, violations)
        request._audit_status = 500
        return (
            jsonify(
                {
                    "error": "Internal schema validation failed.",
                    "violations": violations,
                }
            ),
            500,
        )

    logger.info(
        "ASK  status=%s  top_score=%.3f  method=%s  latency=%.0fms",
        result.get("status"),
        result.get("top_score", 0.0),
        generation_method,
        latency_ms,
    )
    return jsonify(result), 200


@app.post("/api/retrieve")
def retrieve():
    """
    Return raw retrieved chunks for transparency / debugging.

    Body (JSON):
        question     str   required
        k            int   optional (default 3)
        use_reranker bool  optional (default false)
    """
    body = request.get_json(silent=True)
    options, err = _parse_request_options(body)
    if err:
        request._audit_status = 400
        return jsonify({"error": err}), 400
    question, k, use_reranker = options
    request_id = getattr(request, "request_id", "unknown")
    t_start = time.perf_counter()

    try:
        vectordb = get_vectordb()
        chunks = retrieve_with_citation(
            vectordb, question, k=k, use_reranker=use_reranker
        )
    except Exception as exc:
        logger.error("retrieve_with_citation raised: %s", exc)
        request._audit_status = 500
        return jsonify({"error": "Internal retrieval error."}), 500

    latency_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "RETRIEVE request_id=%s question_length=%d chunks=%d latency=%.0fms",
        request_id,
        len(question),
        len(chunks),
        latency_ms,
    )
    return jsonify({"chunks": chunks, "count": len(chunks)}), 200


# -- Error handlers ------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(413)
def request_too_large(e):
    return jsonify({"error": "Request body is too large."}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error."}), 500


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    debug_mode = os.environ.get("APP_DEBUG", "0") == "1"
    port = int(os.environ.get("FLASK_PORT", os.environ.get("APP_PORT", 5000)))
    logger.info("Starting Clinical API on port %d (debug=%s)", port, debug_mode)
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
