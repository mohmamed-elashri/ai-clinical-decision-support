"""CLI for grounded generation. Supports simulation and live modes.

Usage:
    python cli\\grounded_cli.py --mode simulate --question "..."
    python cli\\grounded_cli.py --mode live --question "..."

Live mode expects NV_API_KEY and optional NV_LLM_ENDPOINT environment variables.
"""
import argparse
import json
import os

from generation import answer_question
from llm_client import validate_clinical_response
from query import load_index
from retrieval import retrieve_with_citation


def main():
    parser = argparse.ArgumentParser(description="Clinical Decision Support Grounded CLI")
    parser.add_argument("--mode", choices=["simulate", "live"], default="simulate")
    parser.add_argument("--question", required=True, help="Clinical question to query")
    parser.add_argument("--top_k", type=int, default=3, help="Top K passages to retrieve")
    args = parser.parse_args()

    vectordb = load_index()

    if args.mode == "simulate":
        retrieved = retrieve_with_citation(vectordb, args.question, k=args.top_k)
        top_score = float(retrieved[0].get("dense_score", retrieved[0].get("score", 0.0))) if retrieved else 0.0
        simulated = {
            "status": "grounded" if retrieved else "abstain",
            "answer": (
                "Simulation mode returns retrieved evidence only; it does not fabricate clinical recommendations."
                if retrieved
                else ""
            ),
            "evidence": [chunk.get("text", "")[:300].strip() for chunk in retrieved[:3] if chunk.get("text")],
            "citations": [
                {
                    "document_name": chunk.get("document_name"),
                    "page_number": chunk.get("page_number"),
                    "chunk_id": chunk.get("chunk_id"),
                }
                for chunk in retrieved[:3]
            ],
            "confidence": retrieved[0].get("confidence", "uncertain") if retrieved else "insufficient",
            "top_score": round(top_score, 3),
            "generation_method": "extractive_fallback" if retrieved else "none",
            "retrieved_chunks": retrieved[:3],
        }
        print(json.dumps(simulated, indent=2))
    else:
        resp = answer_question(vectordb, args.question, k=args.top_k)
        is_valid, errors = validate_clinical_response(resp)
        print(json.dumps(resp, indent=2))
        if not is_valid:
            print(f"\n[WARNING] Schema validation warnings: {errors}")


if __name__ == "__main__":
    main()
