"""
run_all.py — Master End-to-End Verification & Reporting Script
AI Clinical Decision Support Lite Hackathon · Insight AI

Usage:
    python run_all.py
"""

import os
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

print()
print("=" * 80)
print("  AI Clinical Decision Support — Master Run (End-to-End)")
print("=" * 80)

PYTHON = sys.executable
PASS = "OK"
FAIL = "FAIL"
results = {}

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"


def run(label, script):
    print(f"\n{'-' * 60}")
    print(f"  Running: {label}")
    print(f"{'-' * 60}")
    t0 = time.time()
    res = subprocess.run([PYTHON, script], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    elapsed = time.time() - t0
    if res.returncode == 0:
        print(res.stdout.strip())
        results[label] = (PASS, f"{elapsed:.1f}s")
    else:
        if res.stderr.strip():
            print(f"STDERR:\n{res.stderr.strip()}")
        if res.stdout.strip():
            print(f"STDOUT:\n{res.stdout.strip()}")
        results[label] = (FAIL, f"{elapsed:.1f}s")
    return res.returncode == 0


STEPS = [
    ("Step 1: Build Vector Index", "ingest.py"),
    ("Step 2: Evaluate Top-K Retrieval", "evaluate_retrieval.py"),
    ("Step 3: Embedding Model Benchmark", "evaluate_embeddings.py"),
    ("Step 4: Architecture Benchmark", "evaluate_retrieval_architectures.py"),
    ("Step 5: Robustness & Anti-Overfitting Test", "evaluate_robustness.py"),
    ("Step 6: Day 1 DoD Verification", "verify_dod.py"),
    ("Step 7: Day 2 DoD Verification", "verify_day2_dod.py"),
    ("Step 8: Day 3 Grounded Generation & Abstain Guard", "verify_day3_dod.py"),
]

all_passed = True
for label, script in STEPS:
    if not run(label, script):
        all_passed = False

print()
print("=" * 80)
print("  MASTER RUN SUMMARY")
print("=" * 80)
for label, (status, duration) in results.items():
    print(f"  [{status}] {label:<55} [{duration}]")
    if status == FAIL:
        all_passed = False

print()
if all_passed:
    print("  ALL STEPS PASSED — Project is verified and submission-ready.")
else:
    print("  Some steps failed — review the output above before submitting.")
print("=" * 80)
print()
sys.exit(0 if all_passed else 1)
