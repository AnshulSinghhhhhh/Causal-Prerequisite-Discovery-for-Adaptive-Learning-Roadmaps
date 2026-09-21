"""Run full E2E walkthrough test for Machine Learning with hardened S0/S6 pipeline."""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_walkthrough import run_demo

print("Starting Machine Learning Walkthrough with hardened pipeline...")
t0 = time.time()
report_path = run_demo("Machine Learning", time_budget=300)
elapsed = time.time() - t0
print(f"\n[DONE] Machine Learning walkthrough completed in {elapsed:.2f}s!")
print(f"Report saved to: {report_path}")
