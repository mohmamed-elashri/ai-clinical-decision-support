"""Thirty-query retrieval benchmark assembled from verified evaluation labels.

The positive questions reuse the existing primary and paraphrase sets, whose
expected pages were already reviewed against the indexed WHO PDFs. Negative
questions are out-of-scope controls and must not be used to measure recall.
"""

from evaluation_set import EVAL_SET, POSITIVE_EVAL_SET
from evaluation_set_robustness import HARD_NEGATIVE_SET, PARAPHRASE_EVAL_SET

EXTRA_NEGATIVE_SET = [
    {
        "question": "What is the recommended chemotherapy regimen for breast cancer?",
        "expected_document": None,
        "expected_pages": [],
        "max_top_score": 0.60,
        "difficulty": "negative",
        "notes": "Oncology question outside the hypertension corpus",
    },
    {
        "question": "How should an acute ischemic stroke be managed in the emergency department?",
        "expected_document": None,
        "expected_pages": [],
        "max_top_score": 0.65,
        "difficulty": "negative",
        "notes": "Neurology question outside the hypertension corpus",
    },
]

PRIMARY_POSITIVE_SET = POSITIVE_EVAL_SET
PARAPHRASE_POSITIVE_SET = PARAPHRASE_EVAL_SET
NEGATIVE_SET = [item for item in EVAL_SET if not item.get("expected_document")]
NEGATIVE_SET = NEGATIVE_SET + HARD_NEGATIVE_SET + EXTRA_NEGATIVE_SET

BENCHMARK_30 = (
    PRIMARY_POSITIVE_SET
    + PARAPHRASE_POSITIVE_SET
    + NEGATIVE_SET
)

assert len(PRIMARY_POSITIVE_SET) == 10
assert len(PARAPHRASE_POSITIVE_SET) == 10
assert len(NEGATIVE_SET) == 10
assert len(BENCHMARK_30) == 30
