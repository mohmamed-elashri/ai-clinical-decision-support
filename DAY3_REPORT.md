# AI Clinical Decision Support Lite — Day 3 Report
## Grounded Generation, Abstention, and Evidence-Based Answers

**Date:** 2026-08-18  
**Goal:** Extend the project from retrieval-only to grounded answer generation with abstention and citation support.

---

## Executive Summary

Day 3 completes the clinical RAG pipeline by turning retrieved evidence into a final answer that is:

- grounded in the retrieved guideline chunks,
- safe when evidence is weak or out of scope,
- traceable to the exact source document and page,
- ready for optional reranking in harder paraphrased queries.

The system now answers only when the retrieved evidence is strong enough, and otherwise refuses to answer instead of guessing.

---

## What Was Added in Day 3

### 1. Grounded generation
The new `generation.py` module builds answers from retrieved evidence instead of generating a free-form response from memory.

It performs the following:

- accepts the medical question,
- retrieves the best evidence chunks,
- selects the strongest relevant sentence or chunk,
- returns an answer based only on the retrieved content.

### 2. Abstention guard
The model checks the top retrieval score against the threshold:

- `OUT_OF_SCOPE_MAX_SCORE = 0.65`

If the evidence is below this threshold, the system returns an abstain response instead of giving an unsupported answer.

This is essential for safety and for preventing hallucinated clinical answers.

### 3. Citation support
Each answer includes structured citation metadata:

- `document_name`
- `page_number`
- `chunk_id`

This makes the answer explainable and auditable.

### 4. Optional reranker
The `retrieval.py` pipeline supports an optional dense reranker (`DenseRerankRetriever`) for more difficult paraphrased questions.

This improves robustness when the question wording differs from the source guideline wording.

---

## System Flow

The final Day 3 flow is:

`query.py` -> `answer(...)` -> `answer_question(...)` -> `retrieve_with_citation(...)` -> grounded answer + citations

### Decision logic
- If evidence is strong and relevant: return a grounded answer.
- If evidence is weak or out of scope: abstain.
- If the user asks a paraphrased question: optionally use reranking to improve retrieval.

---

## Main Files

- `generation.py` — grounded generation and abstain logic
- `retrieval.py` — retrieval wrapper with reranker support
- `query.py` — end-to-end Day 3 interface
- `verify_day3_dod.py` — Day 3 validation script
- `DAY3_REPORT.md` — project report for Day 3

---

## Validation

The Day 3 verification script checks the following:

1. required files exist,
2. in-scope questions generate grounded answers,
3. answers include citations,
4. weak evidence triggers abstention,
5. out-of-scope questions are refused correctly,
6. paraphrase-style questions remain grounded.

### Verified Results

- Day 1: passed (12/12)
- Day 2: passed (21/21)
- Day 3: passed (10/10)

---

## Example Output

Question:

> What is the target blood pressure for a patient with cardiovascular disease?

Output form:

- Status: `grounded`
- Answer: grounded evidence-based sentence from the retrieved guideline
- Citations:
  - `Guideline for the pharmacological treatment of hypertension in adults.pdf (p.28)`
  - additional relevant pages as needed

---

## Why Day 3 Matters

Before Day 3, the system could retrieve relevant guideline text. After Day 3, it can provide a useful final answer while staying grounded and safe.

This is the key step from:

- retrieval system

to:

- evidence-based clinical answer system

---

## Live LLM Generation & Schema Validation (NVIDIA NIM)

In addition to local simulation, Day 3 includes live LLM inference powered by the NVIDIA NIM API (`meta/llama-3.1-8b-instruct`) using `NV_API_KEY`:

1. **Context-Bound Prompting**: Constructs prompt payloads with retrieved evidence excerpts, document citations, and strict clinical constraints (`prompt/grounding_prompt.txt`).
2. **Schema Enforcement**: Validates JSON responses using `jsonschema.Draft7Validator` against `schema/response_schema.json`.
3. **Refusal & Abstention Guard**:
   - **In-scope query**: `"What is the target blood pressure for patients with cardiovascular disease?"` → returns status=`"grounded"`, confidence=`"confident"`, evidence snippets, and verified guideline citations (WHO Hypertension Guideline p. 28, p. 11).
   - **Out-of-scope query**: `"What is the recommended treatment for spotted fever in alpacas?"` → returns status=`"abstain"`, confidence=`"insufficient"`, empty citations/evidence.
   - **Paraphrased query**: `"What blood pressure target should be used for someone with heart disease?"` → returns status=`"grounded"`, confidence=`"confident"`, evidence, and citations.
4. **Notebook Execution**: All live LLM outputs and schema validations are fully executed and saved into `Task3_Grounded_Generation.ipynb`.

---

## Final Conclusion

Day 3 successfully implements the required roadmap:

1. Grounded generation (both local simulation and live NVIDIA NIM LLM)
2. Abstain guard & safety refusal
3. Structured JSON responses adhering to `schema/response_schema.json`
4. Citation-ready responses
5. Optional reranking for harder queries

The project is now complete and verified through its Day 1, Day 2, and Day 3 checks.
