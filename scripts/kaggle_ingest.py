"""Ingest CDP results from Kaggle kernel output into Supabase.

Reads the parquet output, upserts d_cdp into edge_scores,
recomputes s_fused for affected rows.
"""

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def ingest(parquet_path: str = ""):
    """Read CDP results and update edge_scores."""
    from supabase import create_client

    sb = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"],
    )

    if not parquet_path:
        parquet_path = str(ARTIFACT_DIR / "cdp_results.parquet")

    df = pd.read_parquet(parquet_path)
    print(f"Ingesting {len(df)} CDP results...")

    for _, row in df.iterrows():
        # Update edge_scores with d_cdp
        sb.table("edge_scores").update({
            "d_cdp": float(row["d_cdp"]),
        }).eq(
            "candidate_edge_id", row["candidate_edge_id"]
        ).eq(
            "model_version_id", row["model_version_id"]
        ).execute()

    # TODO: Recompute s_fused with calibrated alpha, beta, gamma weights
    # For now, s_fused remains the LR score until gamma is calibrated
    print(f"Ingested {len(df)} CDP scores.")
    print("Run score fusion calibration to update s_fused with CDP signal.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    ingest(path)
