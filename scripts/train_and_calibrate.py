"""CLI script to run baseline training, Youden's J calibration, and database registration.

Usage:
    python scripts/train_and_calibrate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.services.module1.train_baseline import train_and_freeze


def main():
    print("=" * 60)
    print("LightGAP Phase P1: Training & Baseline Calibration Pipeline")
    print("=" * 60)

    res = train_and_freeze()
    print("\n[P1 SUCCESS]")
    print(f"  Model Version ID: {res.get('model_version_id')}")
    print(f"  Frozen Threshold: {res.get('frozen_threshold'):.6f}")
    print(f"  Model Path:       {res.get('model_path')}")
    print(f"  Report Path:      {res.get('report_path')}")


if __name__ == "__main__":
    main()
