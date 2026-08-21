"""
evaluation_set_robustness.py — Anti-overfitting validation set

Paraphrased queries use different wording from the source PDFs to detect
literal-match overfitting. Held separately from the primary EVAL_SET so
the main benchmark is not tuned on these questions.
"""

from evaluation_set import GUIDELINE, HEARTS

# Same ground-truth pages as EVAL_SET, but phrased without PDF copy-paste wording.
PARAPHRASE_EVAL_SET = [
    {
        "question": "For adults with established heart disease, what systolic blood pressure goal should treatment target?",
        "expected_document": GUIDELINE,
        "expected_pages": [28],
        "difficulty": "paraphrase",
        "paired_id": "q01_cvd_target",
    },
    {
        "question": "Which classes of drugs should clinicians pick first when starting hypertension pharmacotherapy?",
        "expected_document": GUIDELINE,
        "expected_pages": [10, 32, 33],
        "difficulty": "paraphrase",
        "paired_id": "q02_first_line",
    },
    {
        "question": "At what blood pressure readings should antihypertensive medication be started in adults?",
        "expected_document": GUIDELINE,
        "expected_pages": [9, 19],
        "difficulty": "paraphrase",
        "paired_id": "q03_initiation",
    },
    {
        "question": "Apart from medicines, what behavioral changes does WHO recommend for people with high blood pressure?",
        "expected_document": GUIDELINE,
        "expected_pages": [14, 15],
        "difficulty": "paraphrase",
        "paired_id": "q04_lifestyle",
    },
    {
        "question": "Under what circumstances should two antihypertensive agents be started together rather than one?",
        "expected_document": GUIDELINE,
        "expected_pages": [26, 38, 39],
        "difficulty": "paraphrase",
        "paired_id": "q05_combination",
    },
    {
        "question": "After a patient begins BP medication, when should they come back for a repeat measurement?",
        "expected_document": GUIDELINE,
        "expected_pages": [30, 37, 38],
        "difficulty": "paraphrase",
        "paired_id": "q06_followup",
    },
    {
        "question": "Summarize how the HEARTS model handles hypertension management in front-line primary care settings.",
        "expected_document": HEARTS,
        "expected_pages": [1, 3, 11],
        "difficulty": "paraphrase",
        "paired_id": "q07_hearts_overview",
    },
    {
        "question": "How should a clinic organize its workflow to deliver HEARTS hypertension services?",
        "expected_document": HEARTS,
        "expected_pages": [11, 28],
        "difficulty": "paraphrase",
        "paired_id": "q08_clinic_org",
    },
    {
        "question": "What step-by-step drug escalation pathway does HEARTS propose for primary care titration?",
        "expected_document": HEARTS,
        "expected_pages": [1, 3, 8, 11, 12, 13, 17, 18, 19, 29],
        "difficulty": "paraphrase",
        "paired_id": "q09_titration",
    },
    {
        "question": "How are cardiovascular risk scores woven into HEARTS hypertension care pathways?",
        "expected_document": HEARTS,
        "expected_pages": [1, 3, 4, 5, 9, 11, 13, 14],
        "difficulty": "paraphrase",
        "paired_id": "q10_risk_integration",
    },
]

# Hard negatives: plausible clinical questions with no answer in the hypertension corpus.
HARD_NEGATIVE_SET = [
    {
        "question": "What insulin regimen is recommended for newly diagnosed type 1 diabetes?",
        "expected_document": None,
        "expected_pages": [],
        "max_top_score": 0.65,
        "notes": "Endocrine — not in hypertension index",
    },
    {
        "question": "What is the WHO-recommended COVID-19 booster interval for immunocompromised adults?",
        "expected_document": None,
        "expected_pages": [],
        "max_top_score": 0.65,
        "notes": "Infectious disease — not in corpus",
    },
    {
        "question": "What surgical approach is preferred for acute appendicitis in pregnancy?",
        "expected_document": None,
        "expected_pages": [],
        "max_top_score": 0.60,
        "notes": "Surgery — unrelated domain",
    },
    {
        "question": "Which inhaled corticosteroid dose should be used for moderate persistent asthma in children?",
        "expected_document": None,
        "expected_pages": [],
        "max_top_score": 0.60,
        "notes": "Pediatric respiratory — unrelated",
    },
    {
        "question": "What is the first-line antidepressant for major depressive disorder in elderly patients?",
        "expected_document": None,
        "expected_pages": [],
        "max_top_score": 0.60,
        "notes": "Psychiatry — unrelated",
    },
    {
        "question": "How should chronic kidney disease stage 4 be managed with dialysis planning?",
        "expected_document": None,
        "expected_pages": [],
        "max_top_score": 0.65,
        "notes": "Nephrology — may share CVD vocabulary; stricter abstain test",
    },
]
