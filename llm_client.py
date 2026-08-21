"""Centralized client for LLM calls (NVIDIA NIM Chat Completions API) and strict schema parsing."""

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

import jsonschema
import requests

logger = logging.getLogger(__name__)

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema", "response_schema.json")
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt", "grounding_prompt.txt")

# Module-level caches — populated on first use, reused on subsequent calls.
_schema_cache: Optional[Dict[str, Any]] = None
_validator_cache: Optional[jsonschema.Draft7Validator] = None
_prompt_cache: Optional[str] = None


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def load_response_schema() -> Dict[str, Any]:
    global _schema_cache, _validator_cache
    if _schema_cache is None:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            _schema_cache = json.load(f)
        _validator_cache = jsonschema.Draft7Validator(_schema_cache)
    return _schema_cache


def get_validator() -> jsonschema.Draft7Validator:
    load_response_schema()
    return _validator_cache  # type: ignore[return-value]


def validate_clinical_response(response_dict: Dict[str, Any]) -> Tuple[bool, list]:
    validator = get_validator()
    errors = sorted(validator.iter_errors(response_dict), key=lambda e: str(e.path))
    return len(errors) == 0, [err.message for err in errors]


def verify_response_provenance(response_dict: Dict[str, Any], retrieved_chunks: list) -> list[str]:
    """Return provenance errors for citations and evidence in a model response."""
    if response_dict.get("status") != "grounded":
        return []

    sources = {}
    source_locations = set()
    source_text = []
    for chunk in retrieved_chunks:
        chunk_id = chunk.get("chunk_id")
        location = (chunk.get("document_name"), str(chunk.get("page_number")))
        if chunk_id:
            sources[str(chunk_id)] = location
        source_locations.add(location)
        source_text.append(_normalize_evidence_text(chunk.get("text", "")))

    errors = []
    citations = response_dict.get("citations", [])
    if not citations:
        errors.append("grounded response must include at least one citation")
    for citation in citations:
        citation_id = citation.get("chunk_id")
        location = (
            citation.get("document_name"),
            str(citation.get("page_number")),
        )
        if citation_id:
            if str(citation_id) not in sources:
                errors.append(f"citation chunk_id is not in retrieved evidence: {citation_id}")
        elif location not in source_locations:
            errors.append("citation document/page is not in retrieved evidence")

    evidence = response_dict.get("evidence", [])
    if not evidence:
        errors.append("grounded response must include evidence snippets")
    for snippet in evidence:
        normalized = _normalize_evidence_text(snippet)
        if normalized and not any(normalized in text for text in source_text):
            errors.append("evidence snippet is not present in retrieved evidence")
    if not response_dict.get("answer", "").strip():
        errors.append("grounded response must include a non-empty answer")
    return errors


def _normalize_evidence_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _load_prompt_template() -> str:
    """Load grounding prompt template from disk once, then serve from cache."""
    global _prompt_cache
    if _prompt_cache is None:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            _prompt_cache = f.read()
        logger.debug("Prompt template loaded from disk: %s", PROMPT_PATH)
    return _prompt_cache


def build_grounded_prompt(question: str, retrieved_chunks: list) -> str:
    """
    Fill the grounding prompt template with retrieved context blocks and the
    user's question.  Uses str.format_map() for safe substitution — avoids
    the fragility of chained str.replace() calls.
    """
    template = _load_prompt_template()

    context_blocks = []
    for r in retrieved_chunks:
        doc = r.get("document_name", "Guideline")
        page = r.get("page_number", "?")
        cid = r.get("chunk_id", "")
        text = r.get("text", "")[:800]
        context_blocks.append(f"SOURCE: {doc} (page {page}, chunk_id: {cid})\n{text}")

    return template.format_map(
        {
            "context_blocks": "\n\n".join(context_blocks),
            "question": question,
        }
    )


# ---------------------------------------------------------------------------
# NIM API call
# ---------------------------------------------------------------------------

def call_nim_chat(
    prompt_text: str,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 30,
) -> str:
    api_key = api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("NV_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set in environment. "
            "Please configure it in .env or your environment variables."
        )

    endpoint = endpoint or os.environ.get(
        "NV_LLM_ENDPOINT", "https://integrate.api.nvidia.com/v1/chat/completions"
    )
    model = model or os.environ.get(
        "NVIDIA_MODEL", os.environ.get("NV_MODEL", "meta/llama-3.1-8b-instruct")
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    system_message = (
        "You are an AI clinical decision support assistant. Return ONLY a single valid JSON object strictly conforming to this schema:\n"
        "- 'status': string, MUST be either 'grounded' or 'abstain'\n"
        "- 'answer': string (the concise recommendation if grounded, or empty string '' if abstain)\n"
        "- 'evidence': array of string snippets from the provided text supporting the answer\n"
        "- 'citations': array of objects, each with 'document_name' (string) and 'page_number' (integer or string), and optional 'chunk_id' (string)\n"
        "- 'confidence': string, MUST be either 'confident', 'uncertain', or 'insufficient'\n"
        "- 'reason': optional string explaining why if abstaining\n"
        "Do NOT include markdown formatting (```json) or extra keys."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    max_attempts = max(1, int(os.environ.get("LLM_MAX_ATTEMPTS", "3")))
    backoff_seconds = max(0.0, float(os.environ.get("LLM_BACKOFF_SECONDS", "0.5")))
    response = None
    for attempt in range(max_attempts):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            break
        except requests.RequestException:
            retryable_response = response is None or response.status_code == 429 or response.status_code >= 500
            if not retryable_response or attempt == max_attempts - 1:
                raise
            time.sleep(backoff_seconds * (2**attempt))

    if response is None:
        raise RuntimeError("LLM request did not return a response")
    response.raise_for_status()
    data = response.json()

    if isinstance(data, dict) and "choices" in data and len(data["choices"]) > 0:
        c = data["choices"][0]
        if isinstance(c, dict) and "message" in c and "content" in c["message"]:
            return c["message"]["content"]
        elif isinstance(c, dict) and "text" in c:
            return c["text"]
    return json.dumps(data)


# ---------------------------------------------------------------------------
# Response parser / normaliser
# ---------------------------------------------------------------------------

def parse_and_clean_json(raw_text: str, top_score: float = 0.0) -> Dict[str, Any]:
    """
    Parse the raw LLM response into a validated, normalised dict.

    Handles:
    - Markdown code fences (```json ... ```)
    - Leading/trailing non-JSON text
    - Loose status / confidence synonyms from the model
    """
    if len(raw_text) > 50_000:
        raise ValueError(
            f"LLM response is suspiciously large ({len(raw_text):,} chars). Refusing to parse."
        )

    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("LLM response must be a JSON object")

    # Normalise status
    status_raw = str(obj.get("status", "")).lower()
    if status_raw in ("grounded", "answer", "success", "ok", "answered"):
        status = "grounded"
    elif status_raw in ("abstain", "refusal", "refuse", "rejected", "insufficient"):
        status = "abstain"
    else:
        status = "grounded" if obj.get("answer") else "abstain"

    # Normalise confidence
    confidence_raw = str(obj.get("confidence", "")).lower()
    if confidence_raw in ("confident", "high", "strong"):
        confidence = "confident"
    elif confidence_raw in ("uncertain", "medium", "moderate"):
        confidence = "uncertain"
    elif confidence_raw in ("insufficient", "low", "none", "unknown"):
        confidence = "insufficient"
    else:
        confidence = "confident" if status == "grounded" else "insufficient"

    # Normalise citations
    valid_citations = []
    for cit in obj.get("citations", []):
        if isinstance(cit, dict) and "document_name" in cit and "page_number" in cit:
            c_dict: Dict[str, Any] = {
                "document_name": str(cit["document_name"]),
                "page_number": (
                    int(cit["page_number"])
                    if str(cit["page_number"]).isdigit()
                    else str(cit["page_number"])
                ),
            }
            if "chunk_id" in cit and cit["chunk_id"]:
                c_dict["chunk_id"] = str(cit["chunk_id"])
            valid_citations.append(c_dict)

    # Normalise evidence
    evidence_list = [
        ev.strip() for ev in obj.get("evidence", []) if isinstance(ev, str) and ev.strip()
    ]

    cleaned: Dict[str, Any] = {
        "status": status,
        "answer": str(obj.get("answer", "")),
        "evidence": evidence_list,
        "citations": valid_citations,
        "confidence": confidence,
    }
    if "reason" in obj and isinstance(obj["reason"], str) and obj["reason"]:
        cleaned["reason"] = obj["reason"]
    if top_score is not None:
        cleaned["top_score"] = round(float(top_score), 3)

    return cleaned
