"""S0.3 — Concept canonicalization (hardened).

Deduplicates extracted concept mentions using a two-tier union-find merge:
  1. Exact match on normalized form → auto-merge
  2. Cosine similarity above calibrated threshold → merge

Normalization (A3) is applied *before* embedding. Union-find (A4) ensures
transitive closure so A≈B and B≈C lands all three in one cluster.
Audit trail (A6) logs all rejections/merges to `rejected_concepts`.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from ...db import table

# Lazy-loaded models
_model: Optional[SentenceTransformer] = None
_nlp = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_nlp():
    """Load spaCy model for lemmatization."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp = False  # sentinel: tried and failed
    return _nlp if _nlp is not False else None


# ---------------------------------------------------------------------------
# A3: Normalize before embedding
# ---------------------------------------------------------------------------

def normalize(name: str) -> str:
    """Normalize a concept name for dedup comparison.

    Steps: lowercase → strip leading determiner → strip possessives
    → collapse whitespace → lemmatize head noun (singular/plural collapse).
    """
    name = name.strip().lower()

    # Strip leading determiner
    name = re.sub(r'^(a|an|the)\s+', '', name)

    # Strip possessives (e.g. "student's" -> "student", "bayes'" -> "bayes")
    name = re.sub(r"'s\b|\b'|(?<=\w)'(?=\s|$)", '', name)

    # Collapse whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    # Lemmatize head noun via spaCy if available
    lemmatized = False
    nlp = _get_nlp()
    if nlp and name:
        try:
            doc = nlp(name)
            if len(doc) > 0:
                last = doc[-1]
                if last.pos_ in ('NOUN', 'PROPN'):
                    prefix = name[:last.idx]
                    name = (prefix + last.lemma_.lower()).strip()
                    lemmatized = True
        except Exception:
            pass

    # Rule-based plural fallback if spaCy was unavailable or didn't handle it
    if not lemmatized and name:
        words = name.split()
        if words:
            last = words[-1]
            if len(last) > 3 and last.endswith('ies'):
                words[-1] = last[:-3] + 'y'
            elif len(last) > 4 and last.endswith(('ses', 'xes', 'ches', 'shes')):
                words[-1] = last[:-2]
            elif len(last) > 3 and last.endswith('s') and not last.endswith(('ss', 'us', 'is')):
                words[-1] = last[:-1]
            name = ' '.join(words)

    return name.strip()



# ---------------------------------------------------------------------------
# A4: Union-find (disjoint-set) data structure
# ---------------------------------------------------------------------------

class UnionFind:
    """Disjoint-set / union-find with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------------------------------------------------------------------------
# A6: Audit trail
# ---------------------------------------------------------------------------

def log_rejected(
    domain_id: str,
    raw_phrase: str,
    stage: str,
    reason: str = "",
    merged_into: Optional[str] = None,
) -> None:
    """Log a rejected concept to the rejected_extractions table."""
    row = {
        "domain_id": domain_id,
        "raw_phrase": raw_phrase,
        "stage": stage,
        "reason": reason or None,
    }
    try:
        table("rejected_extractions").insert(row).execute()
    except Exception:
        pass


def log_rejected_batch(rows: List[Dict]) -> None:
    """Batch insert rejected or merged concepts to rejected_extractions."""
    if not rows:
        return
    for i in range(0, len(rows), 500):
        chunk = rows[i:i + 500]
        try:
            table("rejected_extractions").insert(chunk).execute()
        except Exception:
            pass



# ---------------------------------------------------------------------------
# Core dedup pipeline
# ---------------------------------------------------------------------------

def deduplicate_concepts(
    raw_concepts: List[Dict],
    similarity_threshold: float = 0.92,
    domain_id: Optional[str] = None,
) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    """Merge near-duplicate concepts using union-find over two tiers.

    Tier 1: Exact match on normalized form → auto-merge.
    Tier 2: Cosine similarity above threshold on MiniLM embedding
            of normalized form → merge candidates.

    Returns:
        canonical_concepts: deduplicated concept list
        aliases: list of (canonical_name, alias_name) pairs
    """
    if not raw_concepts:
        return [], []

    from .extract import is_numeric_fragment, is_heading_or_metatoken

    filtered_raw = []
    rejected_structural = []
    for c in raw_concepts:
        c_name = c.get("name", "").strip()
        if is_numeric_fragment(c_name):
            if domain_id:
                rejected_structural.append({
                    "domain_id": domain_id,
                    "raw_phrase": c_name,
                    "stage": "structural_filter",
                    "reason": "numeric_fragment",
                })
            continue
        if is_heading_or_metatoken(c_name):
            if domain_id:
                rejected_structural.append({
                    "domain_id": domain_id,
                    "raw_phrase": c_name,
                    "stage": "structural_filter",
                    "reason": "heading_or_metatoken",
                })
            continue
        filtered_raw.append(c)

    if rejected_structural:
        log_rejected_batch(rejected_structural)

    raw_concepts = filtered_raw
    if not raw_concepts:
        return [], []

    model = _get_model()

    # --- Tier 1: group by normalized form (exact match) ---
    norm_groups: Dict[str, List[int]] = {}  # normalized -> indices
    norms: List[str] = []                   # per-concept normalized form
    for i, c in enumerate(raw_concepts):
        n = normalize(c["name"])
        norms.append(n)
        norm_groups.setdefault(n, []).append(i)

    # Build a representative index list: one per unique normalized form
    unique_norms: List[str] = []
    unique_indices: List[List[int]] = []  # each entry = all raw indices sharing that norm
    norm_to_uid: Dict[str, int] = {}

    for norm, indices in norm_groups.items():
        uid = len(unique_norms)
        norm_to_uid[norm] = uid
        unique_norms.append(norm)
        unique_indices.append(indices)

    n_unique = len(unique_norms)
    if n_unique <= 1:
        # All concepts share one normalized form
        best = _pick_canonical(raw_concepts, [list(range(len(raw_concepts)))])[0]
        aliases = [(best["name"], c["name"]) for c in raw_concepts if c["name"] != best["name"]]
        return [best], aliases

    # --- Tier 2: cosine similarity on normalized embeddings ---
    embeddings = model.encode(unique_norms, normalize_embeddings=True)
    sim_matrix = embeddings @ embeddings.T

    uf = UnionFind(n_unique)

    # Exact-norm matches already share a uid, so they're auto-merged.
    # Now merge by cosine similarity:
    for i in range(n_unique):
        for j in range(i + 1, n_unique):
            if sim_matrix[i, j] >= similarity_threshold:
                uf.union(i, j)

    # --- Collect merged clusters ---
    clusters: Dict[int, List[int]] = {}  # root_uid -> [uid, ...]
    for uid in range(n_unique):
        root = uf.find(uid)
        clusters.setdefault(root, []).append(uid)

    # --- Pick canonical name per cluster ---
    canonical: List[Dict] = []
    aliases: List[Tuple[str, str]] = []
    rejected_to_log: List[Dict] = []

    for root, uids in clusters.items():
        # Gather all raw concepts in this cluster
        cluster_raw_indices = []
        for uid in uids:
            cluster_raw_indices.extend(unique_indices[uid])

        cluster_concepts = [raw_concepts[i] for i in cluster_raw_indices]
        best = _select_canonical_name(cluster_concepts)
        canonical.append(best)

        # Record aliases and log merges (avoid duplicate alias entries per cluster)
        seen_cluster_aliases = set()
        for c in cluster_concepts:
            c_name = c["name"]
            if c_name != best["name"] and c_name not in seen_cluster_aliases:
                seen_cluster_aliases.add(c_name)
                aliases.append((best["name"], c_name))
                if domain_id:
                    rejected_to_log.append({
                        "domain_id": domain_id,
                        "raw_phrase": c_name,
                        "stage": "merged_duplicate",
                        "reason": f"Merged into '{best['name']}'",
                    })

    if domain_id and rejected_to_log:
        log_rejected_batch(rejected_to_log)

    return canonical, aliases



def _select_canonical_name(cluster_concepts: List[Dict]) -> Dict:
    """Select the best canonical name from a cluster of merged concepts.

    Priority:
      1. Form matching a Wikipedia article/redirect title (has source_doc_id
         from a wikipedia source)
      2. Most frequently occurring raw surface form
      3. Shortest well-formed form

    The selected concept inherits the best available definition.
    """
    if len(cluster_concepts) == 1:
        return cluster_concepts[0]

    # Count surface-form frequency
    name_counts = Counter(c["name"] for c in cluster_concepts)

    # Score each candidate
    scored = []
    for c in cluster_concepts:
        name = c["name"]
        # Priority 1: Wikipedia source (source_doc_id present and non-empty)
        wiki_bonus = 1000 if c.get("source_doc_id") else 0
        # Priority 2: Has non-empty definition (gives priority to LLM-defined concepts over spaCy phrases)
        def_bonus = 500 if c.get("definition") else 0
        # Priority 3: frequency
        freq = name_counts[name]
        # Priority 4: prefer shorter (negate length)
        length_score = -len(name)

        scored.append((wiki_bonus, def_bonus, freq, length_score, c))

    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    best = scored[0][4].copy()

    # Inherit definition from any cluster member if best lacks one
    if not best.get("definition"):
        for c in cluster_concepts:
            if c.get("definition"):
                best["definition"] = c["definition"]
                break

    return best


def _pick_canonical(
    raw_concepts: List[Dict],
    groups: List[List[int]],
) -> List[Dict]:
    """Legacy helper: pick best rep per group."""
    result = []
    for indices in groups:
        group = [raw_concepts[i] for i in indices]
        best = max(group, key=lambda x: len(x.get("definition", "")))
        result.append(best)
    return result


def cap_by_relevance(
    concepts: List[Dict],
    goal_concept: str,
    max_concepts: int = 100,
) -> List[Dict]:
    """Keep the most relevant concepts if we have too many.

    Ranks by cosine similarity of each concept's definition
    to the goal concept.
    """
    if len(concepts) <= max_concepts:
        return concepts

    model = _get_model()

    # Embed goal concept
    goal_emb = model.encode([goal_concept], normalize_embeddings=True)[0]

    # Embed each concept's name + definition
    texts = [f"{c['name']}: {c.get('definition', '')}" for c in concepts]
    embeddings = model.encode(texts, normalize_embeddings=True)

    # Rank by similarity to goal
    similarities = embeddings @ goal_emb
    ranked_indices = np.argsort(-similarities)[:max_concepts]

    return [concepts[i] for i in ranked_indices]


def persist_concepts(
    concepts: List[Dict],
    aliases: List[Tuple[str, str]],
    domain_id: str,
) -> List[Dict]:
    """Insert canonical concepts and aliases into the database."""
    model = _get_model()

    # Compute embeddings for all concepts
    texts = [c["name"] for c in concepts]
    embeddings = model.encode(texts, normalize_embeddings=True)

    # Prepare concept rows (deduplicating by canonical_name)
    concept_rows = []
    seen_canon_names = set()
    for c, emb in zip(concepts, embeddings):
        name = c["name"]
        if name in seen_canon_names:
            continue
        seen_canon_names.add(name)
        row = {
            "domain_id": domain_id,
            "canonical_name": name,
            "definition": c.get("definition", ""),
            "source_doc_id": c.get("source_doc_id") or None,
            "embedding": emb.tolist(),
        }
        concept_rows.append(row)

    # Insert concepts in batches of 500
    persisted = []
    for i in range(0, len(concept_rows), 500):
        batch = concept_rows[i:i + 500]
        res = table("concepts").upsert(
            batch,
            on_conflict="domain_id,canonical_name",
        ).execute()
        persisted.extend(res.data or [])

    # Build name->id mapping for aliases
    name_to_id = {r["canonical_name"]: r["id"] for r in persisted}

    # Insert aliases (deduplicating by (concept_id, alias))
    alias_rows = []
    seen_aliases = set()
    for canon_name, alias_name in aliases:
        concept_id = name_to_id.get(canon_name)
        if concept_id:
            key = (concept_id, alias_name)
            if key not in seen_aliases:
                seen_aliases.add(key)
                alias_rows.append({
                    "concept_id": concept_id,
                    "alias": alias_name,
                })

    if alias_rows:
        for i in range(0, len(alias_rows), 500):
            batch = alias_rows[i:i + 500]
            table("concept_aliases").upsert(
                batch,
                on_conflict="concept_id,alias",
            ).execute()

    # A6: Update merged_into in rejected_concepts now that we have IDs
    for canon_name, alias_name in aliases:
        concept_id = name_to_id.get(canon_name)
        if concept_id:
            try:
                table("rejected_concepts").update({
                    "merged_into": concept_id,
                }).eq("domain_id", domain_id).eq(
                    "raw_phrase", alias_name,
                ).eq("stage", "merged_duplicate").execute()
            except Exception:
                pass

    return persisted


def canonicalize(
    raw_concepts: List[Dict],
    goal_concept: str,
    domain_id: str,
    max_concepts: int = 100,
    similarity_threshold: float = 0.92,
) -> List[Dict]:
    """Full canonicalization pipeline: dedupe → cap → persist."""
    # Deduplicate with union-find (A3 + A4)
    canonical, aliases = deduplicate_concepts(
        raw_concepts,
        similarity_threshold=similarity_threshold,
        domain_id=domain_id,
    )

    # Cap by relevance if too many
    canonical = cap_by_relevance(canonical, goal_concept, max_concepts)

    # Persist
    persisted = persist_concepts(canonical, aliases, domain_id)

    return persisted
