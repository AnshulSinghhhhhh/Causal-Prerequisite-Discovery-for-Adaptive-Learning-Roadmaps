"""Canonical configuration for LightGAP.

A single home for every tunable threshold and model choice so that (a) no
module hard-codes a drifting copy and (b) the non-circular calibration rule
(Section 1.1) has one obvious place to *freeze* picked values before they are
applied exactly once to the held-out evaluation set.

Calibrated hyper-parameters (``tau_edge``, ``gamma_misconception``, ``alpha``)
are always the result of training-domain cross-validation, never tuned on the
held-out / gold set. The defaults below are the paper's starting points, to be
updated only via ``calibration.py``-driven runs whose outputs are recorded in
``docs/results/``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Module1Config:
    #: Sentence-transformer used to encode concept text (Section 4.1).
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    #: f_dir(u, v) = [h_u || h_v || (h_u - h_v) || (h_u ⊙ h_v)]  => 4*d
    directional_dim: int = 1536
    #: Number of top-k scored edges per node feeding the candidate graph.
    top_k_candidates: int = 10
    #: DirGCN hidden channels (2-layer; keep within Section 7 budget).
    dirgcn_hidden: int = 64
    #: Dropout for DirGCN training.
    dirgcn_dropout: float = 0.3
    #: Ambiguous band [lo, hi] gating the SLM counterfactual probe.
    ambiguous_lo: float = 0.4
    ambiguous_hi: float = 0.7
    #: Calibrated blend weight alpha for the SLM probe logit.
    alpha: float = _env_float("LIGHTGAP_ALPHA", 0.5)
    #: Number of paraphrased templates averaged per counterfactual probe.
    probe_templates: int = 4
    #: SLM used for counterfactual probing (genuine 3B+-class, 4-bit).
    probe_model: str = "Qwen2.5-3B-Instruct"
    #: Random seed for reproducible splits.
    seed: int = int(os.environ.get("LIGHTGAP_SEED", "42"))


@dataclass(frozen=True)
class Module2Config:
    #: Prerequisite edge threshold (paper default 0.65; tune on validation).
    tau_edge: float = _env_float("LIGHTGAP_TAU_EDGE", 0.65)
    #: Whether to apply transitive reduction after cycle pruning (default on).
    transitive_reduce: bool = True


@dataclass(frozen=True)
class Module3Config:
    #: Misconception match cutoff (paper default 0.1; grid-searched jointly
    #: with ``tau_edge`` against the same held-out validation portion).
    gamma_misconception: float = _env_float("LIGHTGAP_GAMMA_MC", 0.1)
    #: Quizzes per concept node (fixed by the paper).
    questions_per_node: int = 3
    #: API cache TTL in seconds (rate limits; never re-query per render).
    cache_ttl_seconds: int = 3600
    #: Maximum content items retained per node after re-ranking.
    max_items_per_node: int = 5
    #: API clients enabled at runtime (empty list = offline / no-op clients).
    enabled_apis: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Module4Config:
    #: Retention decay threshold (R_i < tau_decay => refresher_needed).
    tau_decay: float = _env_float("LIGHTGAP_TAU_DECAY", 0.5)
    #: Scheduler backing the decay model. One of {"sm2", "fsrs"}.
    scheduler: str = "sm2"
    #: SM-2 default ease factor (global cold-start prior seed).
    ease_initial: float = 2.5
    #: FSRS default stability (days) used as the global cold-start prior.
    fsrs_default_stability_days: float = 2.0
    #: Score threshold treated as "success" when updating stability.
    success_score: int = 3
    #: Number of consecutive successful reviews to resolve a remediation node.
    resolution_successes: int = 1


@dataclass(frozen=True)
class BudgetConfig:
    """Section 7 / Section 9 hardware budget, surfaceed so it is enforceable."""

    max_total_footprint_mb: int = 3000  # < 3 GB total inference footprint
    embedding_mb: int = 80
    dirgcn_train_mb: int = 300
    slm_probe_mb: int = 2200  # ~2.2 GB 4-bit 3B-class SLM
    vector_index_mb: int = 50


@dataclass(frozen=True)
class Config:
    """Aggregated, immutable configuration consumed app-wide."""

    module1: Module1Config = field(default_factory=Module1Config)
    module2: Module2Config = field(default_factory=Module2Config)
    module3: Module3Config = field(default_factory=Module3Config)
    module4: Module4Config = field(default_factory=Module4Config)
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    def to_dict(self) -> Dict[str, Dict]:
        return {
            "module1": self.module1.__dict__,
            "module2": self.module2.__dict__,
            "module3": self.module3.__dict__,
            "module4": self.module4.__dict__,
            "budget": self.budget.__dict__,
        }


#: Process-wide singleton. Override individual fields by constructing a new
#: ``Config`` instance (dataclasses are frozen/immutable) or via the
#: ``LIGHTGAP_*`` environment variables read at import time.
config: Config = Config()


def reset_config() -> None:
    """Re-read environment variables (used by tests that exercise env knobs)."""
    Config.__dataclass_fields__  # noqa: B018 -- touch for static checkers
    global config
    config = Config()