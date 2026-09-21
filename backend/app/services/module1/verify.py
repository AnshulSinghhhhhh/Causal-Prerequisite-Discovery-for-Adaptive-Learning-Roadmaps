"""S2 — Directional verification.

For each candidate pair, scores both orientations with the frozen
LR model. Keeps the higher-scoring direction if it clears tau_edge.
Low-margin pairs (bottom quartile) are flagged for S3 (CDP probe).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from ...db import table

_model: Optional[SentenceTransformer] = None
_lr_model = None

DATA_ROOT = Path(__file__).resolve().parents[4] / "data"


def _get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _build_features(h_u: np.ndarray, h_v: np.ndarray) -> np.ndarray:
    """Build asymmetric directional feature vector.
    
    f_dir(u,v) = [h_u || h_v || (h_u - h_v) || (h_u * h_v)]
    
    The asymmetry (difference and product don't commute) is what
    lets a plain logistic regression detect direction.
    """
    return np.concatenate([
        h_u,           # 384-d
        h_v,           # 384-d
        h_u - h_v,     # 384-d (asymmetric)
        h_u * h_v,     # 384-d (asymmetric via interaction with difference)
    ])


def get_frozen_threshold(model_version_id: str) -> float:
    """Read the frozen threshold from the model_versions table."""
    result = table("model_versions").select("frozen_threshold").eq(
        "id", model_version_id
    ).single().execute()
    return float(result.data["frozen_threshold"])


def load_lr_model(model_version_id: Optional[str] = None):
    """Load the trained logistic regression model.
    
    Tries to load from the database artifact URI first,
    falls back to local pkl files.
    """
    global _lr_model
    if _lr_model is not None:
        return _lr_model
    
    # Try local model files
    for pkl_path in [
        DATA_ROOT / "models" / "logistic_baseline.pkl",
        DATA_ROOT / "models" / "model_al_cpl.pkl",
    ]:
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                _lr_model = pickle.load(f)
            return _lr_model
    
    raise FileNotFoundError(
        "No trained LR model found. Run scripts/train_and_calibrate.py first."
    )


def verify_candidates(
    candidates: List[Dict],
    concepts: List[Dict],
    model_version_id: str,
    tau_edge: Optional[float] = None,
) -> List[Dict]:
    """Score both orientations of each candidate pair.
    
    For each candidate:
    1. Embed both concepts
    2. Build f_dir in both directions
    3. Score with LR model
    4. Keep the higher-scoring direction if it clears tau_edge
    5. Compute margin = |forward - reverse|
    
    Returns list of edge_score dicts ready for DB insertion.
    """
    embedder = _get_embedder()
    lr_model = load_lr_model()
    
    if tau_edge is None:
        tau_edge = get_frozen_threshold(model_version_id)
    
    # Build concept embedding lookup
    concept_texts = {c["id"]: c["canonical_name"] for c in concepts}
    all_ids = list(set(
        [c["src_id"] for c in candidates] + [c["dst_id"] for c in candidates]
    ))
    all_texts = [concept_texts.get(cid, cid) for cid in all_ids]
    all_embeddings = embedder.encode(all_texts, normalize_embeddings=True)
    emb_lookup = dict(zip(all_ids, all_embeddings))
    
    edge_scores = []
    for cand in candidates:
        src_id = cand["src_id"]
        dst_id = cand["dst_id"]
        cand_id = cand.get("id", "")
        if not cand_id:
            continue
        
        h_src = emb_lookup.get(src_id)
        h_dst = emb_lookup.get(dst_id)
        
        if h_src is None or h_dst is None:
            continue
        
        # Score both directions
        feat_forward = _build_features(h_src, h_dst).reshape(1, -1)
        feat_reverse = _build_features(h_dst, h_src).reshape(1, -1)
        
        s_forward = float(lr_model.predict_proba(feat_forward)[0, 1])
        s_reverse = float(lr_model.predict_proba(feat_reverse)[0, 1])
        
        margin = abs(s_forward - s_reverse)
        admitted = max(s_forward, s_reverse) >= tau_edge
        
        edge_scores.append({
            "candidate_edge_id": cand_id,
            "model_version_id": model_version_id,
            "s_lr_forward": s_forward,
            "s_lr_reverse": s_reverse,
            "admitted": admitted,
            "s_fused": max(s_forward, s_reverse),  # initially just LR score
        })

    
    return edge_scores


def persist_edge_scores(edge_scores: List[Dict], chunk_size: int = 500) -> List[Dict]:
    """Insert edge scores into the database in chunks."""
    if not edge_scores:
        return []
    
    all_data = []
    for i in range(0, len(edge_scores), chunk_size):
        chunk = edge_scores[i:i + chunk_size]
        result = table("edge_scores").upsert(
            chunk,
            on_conflict="candidate_edge_id,model_version_id",
        ).execute()
        if result.data:
            all_data.extend(result.data)
    return all_data


def get_low_margin_edges(
    model_version_id: str,
    percentile: float = 0.25,
) -> List[Dict]:
    """Get bottom-quartile margin edges for S3 CDP probing."""
    from ...db import fetch_all
    query = table("edge_scores").select("*").eq(
        "model_version_id", model_version_id
    ).is_("d_cdp", "null").order("margin")
    edges = fetch_all(query)
    
    if not edges:
        return []
    
    cutoff = int(len(edges) * percentile)
    return edges[:max(cutoff, 1)]

