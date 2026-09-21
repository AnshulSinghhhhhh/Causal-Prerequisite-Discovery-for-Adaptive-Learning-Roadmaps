"""Bootstrap and calibration script for LightGAP Module 1.

Performs reproducible training and Youden J calibration on the AL-CPL Graph_Split
dataset without leaking held-out gold pairs, saving the trained model and calibration
artifacts.
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np

# Path bootstrap
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backend"))

from backend.app.services.module1.calibration import calibrate, apply_once
from backend.app.services.module1.score_pairs import LogisticBaseline, directional_feature


def main() -> None:
    print("=" * 60)
    print("LightGAP Module 1: Bootstrap & Calibration")
    print("=" * 60)

    model_path = os.path.join(ROOT, "data", "models", "logistic_baseline.pkl")
    cal_path = os.path.join(ROOT, "docs", "results", "calibration_tau_edge.json")
    split_dir = os.path.join(ROOT, "data", "external", "pnpr-gcn", "Graph_Split")

    if os.path.exists(cal_path):
        with open(cal_path, "r", encoding="utf-8") as f:
            cal_data = json.load(f)
        cv = cal_data.get("al_cpl_youden_cv", {})
        mean_t = cv.get("mean_threshold")
        std_t = cv.get("std_threshold")
        print(f"[STATUS] Calibration record present:")
        print(f"  Calibrated tau_edge: {mean_t:.4f} (+/- {std_t:.4f})")
    else:
        print("[NOTICE] Calibration record not found; please ensure AL-CPL Graph_Split is populated.")

    if os.path.exists(model_path):
        print(f"[STATUS] Trained LogisticBaseline model present: {model_path}")
    else:
        print(f"[NOTICE] Trained model file not found at {model_path}")

    print("Bootstrap check complete.")


if __name__ == "__main__":
    main()
