"""S1 — Multi-signal candidate generation.

The highest-value stage in LightGAP. Generates candidate prerequisite
pairs using four deterministic signals (no model calls), tuned for
recall. Precision is S2's job.

Signals:
- s_order:      fraction of documents where A precedes B
- s_defmention: does definition(B) reference A?
- s_cooc:       normalized PMI within same section
- s_sim:        MiniLM cosine (pruning only — never used in s_corr)
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from ...db import table

load_dotenv()

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_groq_client():
    from groq import Groq
    return Groq(api_key=os.environ.get("GROQ_API_KEY", ""))


def compute_s_order(
    concepts: List[Dict],
    documents: List[Dict],
    syllabus_weight: float = 3.0,
    default_weight: float = 1.0,
) -> Dict[Tuple[str, str], float]:
    """Compute s_order: source-weighted fraction of documents where A precedes B.
    
    A syllabus-sourced ordering observation counts for more than a Wikipedia-prose
    one (3x syllabus vs 1x wikipedia/other), reflecting actual pedagogical sequencing
    decisions by instructors over incidental prose proximity.
    """
    # Build concept name -> id mapping
    name_to_id = {}
    for c in concepts:
        name_to_id[c["canonical_name"].lower()] = c["id"]
        # Include aliases if available
        for alias in c.get("aliases", []):
            name_to_id[alias.lower()] = c["id"]
    
    # For each document, find first occurrence position of each concept
    pair_order_counts: Dict[Tuple[str, str], float] = defaultdict(float)
    pair_coappear_counts: Dict[Tuple[str, str], float] = defaultdict(float)
    
    for doc in documents:
        text = doc.get("raw_text", "").lower()
        if not text:
            continue
        
        doc_type = doc.get("source_type", "wikipedia")
        doc_weight = syllabus_weight if doc_type == "syllabus" else default_weight
        
        # Find first position of each concept in this doc
        positions: Dict[str, int] = {}
        for name, cid in name_to_id.items():
            pos = text.find(name)
            if pos >= 0:
                if cid not in positions or pos < positions[cid]:
                    positions[cid] = pos
        
        # Count ordered co-occurrences
        concept_ids = list(positions.keys())
        for i, a_id in enumerate(concept_ids):
            for j, b_id in enumerate(concept_ids):
                if a_id == b_id:
                    continue
                pair_coappear_counts[(a_id, b_id)] += doc_weight
                if positions[a_id] < positions[b_id]:
                    pair_order_counts[(a_id, b_id)] += doc_weight
    
    # Compute fraction
    scores: Dict[Tuple[str, str], float] = {}
    for pair, coappear in pair_coappear_counts.items():
        if coappear > 0:
            scores[pair] = pair_order_counts.get(pair, 0.0) / coappear
    
    return scores


def compute_s_defmention(
    concepts: List[Dict],
) -> Dict[Tuple[str, str], float]:
    """Compute s_defmention: does definition(B) reference A?
    
    For each pair (A, B), checks if A's canonical name or aliases
    appear in B's definition. Weighted by position (earlier = stronger).
    Strongest single cue in the prerequisite literature.
    """
    scores: Dict[Tuple[str, str], float] = {}
    
    # Build lookup: concept_id -> {canonical_name, aliases}
    concept_names: Dict[str, List[str]] = {}
    for c in concepts:
        names = [c["canonical_name"].lower()]
        for alias in c.get("aliases", []):
            names.append(alias.lower())
        concept_names[c["id"]] = names
    
    for b in concepts:
        b_def = (b.get("definition") or "").lower()
        if not b_def:
            continue
        
        for a in concepts:
            if a["id"] == b["id"]:
                continue
            
            # Check if any of A's names appear in B's definition
            best_score = 0.0
            for name in concept_names[a["id"]]:
                pos = b_def.find(name)
                if pos >= 0:
                    # Weight by position: earlier mention = stronger signal
                    position_weight = 1.0 - (pos / max(len(b_def), 1))
                    best_score = max(best_score, 0.5 + 0.5 * position_weight)
            
            if best_score > 0:
                scores[(a["id"], b["id"])] = best_score
    
    return scores


def compute_s_cooc(
    concepts: List[Dict],
    documents: List[Dict],
    section_sep: str = "\n\n",
) -> Dict[Tuple[str, str], float]:
    """Compute s_cooc: normalized PMI within same paragraph/section.
    
    Measures how often A and B co-occur in the same section,
    normalized to [0, 1]. Recall-oriented.
    """
    # Split documents into sections
    total_sections = 0
    concept_section_count: Dict[str, int] = defaultdict(int)
    pair_section_count: Dict[Tuple[str, str], int] = defaultdict(int)
    
    name_to_id = {}
    for c in concepts:
        name_to_id[c["canonical_name"].lower()] = c["id"]
    
    for doc in documents:
        text = doc.get("raw_text", "")
        sections = text.split(section_sep)
        
        for section in sections:
            if len(section.strip()) < 20:
                continue
            total_sections += 1
            section_lower = section.lower()
            
            # Find which concepts appear in this section
            present = set()
            for name, cid in name_to_id.items():
                if name in section_lower:
                    present.add(cid)
            
            for cid in present:
                concept_section_count[cid] += 1
            
            # Count co-occurrences
            present_list = list(present)
            for i, a in enumerate(present_list):
                for j, b in enumerate(present_list):
                    if a != b:
                        pair_section_count[(a, b)] += 1
    
    # Compute normalized PMI
    scores: Dict[Tuple[str, str], float] = {}
    if total_sections == 0:
        return scores
    
    for (a, b), cooc in pair_section_count.items():
        p_ab = cooc / total_sections
        p_a = concept_section_count.get(a, 0) / total_sections
        p_b = concept_section_count.get(b, 0) / total_sections
        
        if p_a > 0 and p_b > 0 and p_ab > 0:
            pmi = math.log(p_ab / (p_a * p_b))
            # Normalize: NPMI = PMI / -log(p_ab)
            npmi = pmi / (-math.log(p_ab))
            # Clamp to [0, 1]
            scores[(a, b)] = max(0.0, min(1.0, (npmi + 1) / 2))
    
    return scores


def compute_s_sim(
    concepts: List[Dict],
) -> Dict[Tuple[str, str], float]:
    """Compute s_sim: MiniLM cosine similarity.
    
    PRUNING ONLY — discard pairs below a low floor.
    Never used to score direction (similarity has no directional info).
    Do not let this leak into s_corr's weighting.
    """
    model = _get_model()
    
    texts = [f"{c['canonical_name']}: {c.get('definition', '')}" for c in concepts]
    embeddings = model.encode(texts, normalize_embeddings=True)
    
    sim_matrix = embeddings @ embeddings.T
    
    scores: Dict[Tuple[str, str], float] = {}
    for i in range(len(concepts)):
        for j in range(len(concepts)):
            if i != j:
                scores[(concepts[i]["id"], concepts[j]["id"])] = float(sim_matrix[i, j])
    
    return scores


def compute_s_llm_plaus(
    pairs: List[Tuple[str, str]],
    concepts: List[Dict],
    batch_size: int = 15,
    max_eval_pairs: int = 75,
) -> Dict[Tuple[str, str], float]:
    """Compute s_llm_plaus: LLM plausibility score for candidate pairs.
    
    Evaluates prerequisite plausibility (0.0 to 1.0) via batched Groq calls.
    Chunked into batches (default 15) with a fixed rubric:
      - 1.0: Essential prerequisite
      - 0.7: Strong supporting prerequisite
      - 0.3: Related topic, caveat, or historical aside
      - 0.0: Unrelated or reversed relationship
    
    To stay strictly within API token quotas, evaluates up to max_eval_pairs
    (defaults to top 75 candidate pairs ranked by heuristic strength).
    Pairs beyond the quota receive a neutral 0.5 prior.
    Falls back gracefully between configured model and groq/compound-mini on rate limits.
    """
    if not pairs:
        return {}
    
    id_to_name = {c["id"]: c["canonical_name"] for c in concepts}
    for c in concepts:
        for alias in c.get("aliases", []):
            if alias not in id_to_name:
                id_to_name[alias] = c["canonical_name"]
    
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        return {pair: 0.5 for pair in pairs}
    
    primary_model = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
    candidate_models = [primary_model]
    for cand in ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "groq/compound-mini"]:
        if cand not in candidate_models:
            candidate_models.append(cand)
    
    client = _get_groq_client()
    scores: Dict[Tuple[str, str], float] = {pair: 0.5 for pair in pairs}
    eval_pairs = pairs[:max_eval_pairs]
    
    for i in range(0, len(eval_pairs), batch_size):
        batch = eval_pairs[i:i + batch_size]
        batch_lines = []
        for j, (a_id, b_id) in enumerate(batch):
            a_name = id_to_name.get(a_id, str(a_id))
            b_name = id_to_name.get(b_id, str(b_id))
            batch_lines.append(f"{j+1}. '{a_name}' -> '{b_name}'")
        
        prompt = f"""You are reviewing candidate prerequisite relationships between pairs of concepts.
For each pair (A -> B), evaluate: does knowing A plausibly help you understand B — as a real prerequisite, not just something commonly discussed nearby?

Answer per item with a plausibility score between 0.0 and 1.0:
- 1.0: Essential prerequisite (B directly builds upon or requires understanding A)
- 0.7: Strong supporting prerequisite (A is a foundational tool or core component of B)
- 0.3: Related topic, caveat, or historical aside (often discussed together, but A is not a prerequisite for B)
- 0.0: Unrelated, or reversed relationship (B is a prerequisite for A, not vice-versa)

Candidate pairs:
{chr(10).join(batch_lines)}

Respond strictly with a JSON array where each element is:
{{"index": <1-based index>, "score": <float between 0.0 and 1.0>}}
"""
        succeeded = False
        for model in candidate_models:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=450,
                    timeout=25.0,
                )
                content = resp.choices[0].message.content.strip()
                if "```" in content:
                    m = re.search(r'```(?:json)?\s*(.+?)```', content, re.DOTALL)
                    content = m.group(1).strip() if m else content
                
                data = json.loads(content)
                verdict_map = {}
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "index" in item and "score" in item:
                            try:
                                verdict_map[int(item["index"])] = float(item["score"])
                            except (ValueError, TypeError):
                                pass
                
                for j, pair in enumerate(batch):
                    s = verdict_map.get(j + 1, 0.5)
                    scores[pair] = max(0.0, min(1.0, float(s)))
                
                time.sleep(0.5)
                succeeded = True
                break
            except Exception as e:
                print(f"[compute_s_llm_plaus] Model {model} attempt failed: {e}")
                continue
        
        if not succeeded:
            for pair in batch:
                scores[pair] = 0.5
    
    return scores


def generate_candidates(
    concepts: List[Dict],
    documents: List[Dict],
    domain_id: str,
    weights: Optional[Dict[str, float]] = None,
    sim_floor: float = 0.15,
    corr_threshold: float = 0.10,
) -> List[Dict]:
    """Generate candidate prerequisite edges using all five signals.
    
    Args:
        concepts: list of concept dicts with id, canonical_name, definition, aliases
        documents: list of document dicts with id, raw_text
        domain_id: the domain UUID
        weights: signal weights for s_corr (calibrated on AL-CPL CV)
        sim_floor: minimum s_sim to keep a pair (pruning threshold)
        corr_threshold: minimum s_corr to admit a candidate (recall-oriented)
    
    Returns:
        List of candidate_edge dicts ready for DB insertion.
    """
    if weights is None:
        weights = {
            "s_order": 0.25,
            "s_defmention": 0.35,
            "s_cooc": 0.10,
            "s_llm_plaus": 0.30,
            # s_sim is NOT in this dict — it's pruning only
        }
    
    # Compute deterministic signals
    s_order = compute_s_order(concepts, documents)
    s_defmention = compute_s_defmention(concepts)
    s_cooc = compute_s_cooc(concepts, documents)
    s_sim = compute_s_sim(concepts)
    
    # Collect all concept ID pairs and deduplicate into unordered pairs (canonical min, max)
    raw_pairs = set()
    raw_pairs.update(s_order.keys())
    raw_pairs.update(s_defmention.keys())
    raw_pairs.update(s_cooc.keys())
    
    unordered_pairs = set()
    for (u, v) in raw_pairs:
        if u != v:
            unordered_pairs.add((min(u, v), max(u, v)))
    
    # Prune pairs below similarity floor before LLM plausibility pass
    surviving_pairs = [pair for pair in unordered_pairs if s_sim.get(pair, 0.0) >= sim_floor]
    
    # Order surviving pairs by preliminary heuristic association strength (max across directions)
    def pair_heuristic(p):
        u, v = p
        h_uv = (
            0.35 * s_defmention.get((u, v), 0.0)
            + 0.25 * s_order.get((u, v), 0.0)
            + 0.10 * s_cooc.get((u, v), 0.0)
        )
        h_vu = (
            0.35 * s_defmention.get((v, u), 0.0)
            + 0.25 * s_order.get((v, u), 0.0)
            + 0.10 * s_cooc.get((v, u), 0.0)
        )
        return max(h_uv, h_vu)

    surviving_pairs.sort(key=pair_heuristic, reverse=True)
    
    # For LLM plausibility pass, orient each pair according to higher preliminary signal
    oriented_pairs_for_llm = []
    for (u, v) in surviving_pairs:
        h_uv = 0.35 * s_defmention.get((u, v), 0.0) + 0.25 * s_order.get((u, v), 0.0)
        h_vu = 0.35 * s_defmention.get((v, u), 0.0) + 0.25 * s_order.get((v, u), 0.0)
        oriented_pairs_for_llm.append((u, v) if h_uv >= h_vu else (v, u))
    
    # Compute fifth signal: s_llm_plaus
    s_llm_plaus_raw = compute_s_llm_plaus(oriented_pairs_for_llm, concepts)
    
    candidates = []
    for pair in surviving_pairs:
        a_id, b_id = pair
        sim = s_sim.get(pair, 0.0)
        
        # Aggregate directional signals to represent the pair's prerequisite candidacy
        s_ord_max = max(s_order.get((a_id, b_id), 0.0), s_order.get((b_id, a_id), 0.0))
        s_def_max = max(s_defmention.get((a_id, b_id), 0.0), s_defmention.get((b_id, a_id), 0.0))
        s_cooc_val = s_cooc.get(pair, s_cooc.get((b_id, a_id), 0.0))
        s_llm_val = max(
            s_llm_plaus_raw.get((a_id, b_id), 0.5),
            s_llm_plaus_raw.get((b_id, a_id), 0.5),
        )
        
        signal_values = {
            "s_order": s_ord_max,
            "s_defmention": s_def_max,
            "s_cooc": s_cooc_val,
            "s_llm_plaus": s_llm_val,
        }
        
        total_weight = sum(weights.values())
        s_corr = sum(
            weights[k] * signal_values[k] for k in weights
        ) / total_weight
        
        # Apply threshold (deliberately low — recall-oriented)
        if s_corr < corr_threshold:
            continue
        
        candidates.append({
            "domain_id": domain_id,
            "src_id": a_id,
            "dst_id": b_id,
            "s_order": signal_values["s_order"],
            "s_cooc": signal_values["s_cooc"],
            "s_defmention": signal_values["s_defmention"],
            "s_sim": sim,
            "s_llm_plaus": signal_values["s_llm_plaus"],
            "s_corr": s_corr,
        })
    
    return candidates



def persist_candidates(candidates: List[Dict], chunk_size: int = 500) -> List[Dict]:
    """Insert candidate edges into the database in chunks."""
    if not candidates:
        return []
    
    all_res = []
    for i in range(0, len(candidates), chunk_size):
        chunk = candidates[i:i + chunk_size]
        try:
            result = table("candidate_edges").upsert(
                chunk,
                on_conflict="src_id,dst_id",
            ).execute()
            if result.data:
                all_res.extend(result.data)
        except Exception as e:
            if "s_llm_plaus" in str(e):
                # Remote schema does not have s_llm_plaus column yet;
                # strip for DB insert while preserving in memory
                stripped = [
                    {k: v for k, v in c.items() if k != "s_llm_plaus"}
                    for c in chunk
                ]
                result = table("candidate_edges").upsert(
                    stripped,
                    on_conflict="src_id,dst_id",
                ).execute()
                res_data = result.data or []
                # Merge DB id into chunk so downstream S2 has candidate_edge_id
                id_map = {(r["src_id"], r["dst_id"]): r["id"] for r in res_data if "id" in r}
                for c in chunk:
                    if (c["src_id"], c["dst_id"]) in id_map:
                        c["id"] = id_map[(c["src_id"], c["dst_id"])]
                all_res.extend([c for c in chunk if "id" in c] or res_data)
            else:
                raise
    return all_res
