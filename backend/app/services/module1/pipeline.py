"""Module 1 -- stage orchestration.

Ties the three stages together into the single pipeline the paper describes:

    text -> embeddings -> stage-1 baseline scores -> candidate graph
          -> stage-2 DirGCN refinement -> stage-3 SLM counterfactual probe
          (ambiguous band only) -> refined P(u -> v)

This is the seam the rest of the system (and the evaluation harness in
``scripts/run_full_pipeline.py``) imports. It does NOT import real dataset
loaders or model weights; those live in scripts, keeping ``app/services``
stable and offline-importable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from .counterfactual_probe import ProbeBackend, probe_ambiguous_edges
from .embed import ConceptEmbedder, build_concept_texts
from .score_pairs import (
    LogisticBaseline,
    ScoreBatch,
    build_candidate_graph,
    score_pairs,
)


@dataclass
class Module1Output:
    """Scored, directed edge list produced by the prerequisite engine."""

    #: Directed pairs in decision order (u, v).
    pairs: List[Tuple[str, str]] = field(default_factory=list)
    #: Final refined stage-3 scores (P(u -> v)).
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Stage-1 baseline scores (pre-GNN, pre-probe) for ablations.
    stage1_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Stage-2 DirGCN scores (post-GNN, pre-probe).
    stage2_scores: Optional[np.ndarray] = None
    #: Raw ascending source->target order preserved from scoring.
    node_order: List[str] = field(default_factory=list)

    def edge_records(self) -> List[dict]:
        out = []
        for (u, v), s in zip(self.pairs, self.scores.tolist()):
            out.append({"source": u, "target": v, "confidence": float(s)})
        return out


class Module1Pipeline:
    """End-to-end prerequisite engine with pluggable optional stages."""

    def __init__(
        self,
        embedder: Optional[ConceptEmbedder] = None,
        stage1: Optional[LogisticBaseline] = None,
        dirgcn: Optional[object] = None,
        probe_backend: Optional[ProbeBackend] = None,
        alpha: Optional[float] = None,
    ) -> None:
        self.embedder = embedder or ConceptEmbedder()
        self.stage1 = stage1
        self.dirgcn = dirgcn           # optional; None skips stage 2
        self.probe_backend = probe_backend  # optional; None skips stage 3
        self.alpha = alpha

    @classmethod
    def create_default(
        cls,
        use_groq_probe: bool = False,
        groq_api_key: Optional[str] = None,
        groq_model: Optional[str] = None,
        use_llamacpp_probe: bool = False,
        llamacpp_model_path: Optional[str] = None,
    ) -> "Module1Pipeline":
        """Factory method to instantiate a pipeline with standard, Groq, or local llama.cpp probe."""
        probe: Optional[ProbeBackend] = None
        if use_llamacpp_probe:
            try:
                from .llamacpp_probe import LocalLlamaCppProbeBackend
                probe = LocalLlamaCppProbeBackend(model_path=llamacpp_model_path)
            except Exception as e:
                logger.warning("[Module1Pipeline] Failed to initialize LocalLlamaCppProbeBackend: %s", e)
                probe = None
        elif use_groq_probe:
            try:
                from .groq_probe import GroqProbeBackend
                kwargs = {}
                if groq_api_key:
                    kwargs["api_key"] = groq_api_key
                if groq_model:
                    kwargs["model"] = groq_model
                probe = GroqProbeBackend(**kwargs)
            except Exception:
                probe = None
        return cls(probe_backend=probe)

    # -- embeddings ----------------------------------------------------- #
    def embed(self, concepts: Sequence[object]) -> Dict[str, np.ndarray]:
        texts = build_concept_texts(concepts)
        result = self.embedder.embed(texts)
        ids = [getattr(c, "id", getattr(c, "name", c)) if not isinstance(c, str)
               else c for c in concepts]
        return {str(i): result.vectors[k] for k, i in enumerate(ids)}

    # -- scoring -------------------------------------------------------- #
    def run(
        self,
        embeddings: Dict[str, np.ndarray],
        pairs: Sequence[Tuple[str, str]],
        stage1: Optional[LogisticBaseline] = None,
    ) -> Module1Output:
        """Score ``pairs`` through stages 1 -> 2 -> 3 and return the result."""
        model = stage1 or self.stage1
        if model is None:
            raise RuntimeError(
                "Module1Pipeline.run requires a trained stage-1 baseline; "
                "call .fit_stage1(...) or pass a fitted LogisticBaseline."
            )
        batch = score_pairs(embeddings, pairs, model)
        s1 = np.asarray(batch.scores, dtype=float)
        s2: Optional[np.ndarray] = None
        if self.dirgcn is not None:
            s2 = self._run_dirgcn(embeddings, pairs, s1)
        current = s2 if s2 is not None else s1
        final = current
        if self.probe_backend is not None:
            final, _ = probe_ambiguous_edges(
                pairs, current, self.probe_backend, alpha=self.alpha
            )
        return Module1Output(
            pairs=list(pairs),
            scores=np.asarray(final, dtype=float),
            stage1_scores=s1,
            stage2_scores=s2,
            node_order=list(embeddings.keys()),
        )

    def _run_dirgcn(
        self, embeddings: Dict[str, np.ndarray],
        pairs: Sequence[Tuple[str, str]], s1: np.ndarray,
    ) -> np.ndarray:
        from .dirgcn import build_edge_index, run_dirgcn

        nodes = list(embeddings.keys())
        idx = {n: i for i, n in enumerate(nodes)}
        # Candidate graph from stage-1 top-k (never edgeless; guards raise).
        cand = build_candidate_graph(
            embeddings, self.stage1, nodes,
        )
        edge_pairs = [(idx[u], idx[v]) for u, v in cand.pairs]
        ei = build_edge_index(edge_pairs).numpy()
        x = np.stack([embeddings[n] for n in nodes])
        pr_pairs = [(idx[u], idx[v]) for u, v in pairs]
        # dirgcn object may expose .state_dict for a trained model.
        state = getattr(self.dirgcn, "state_dict", None)
        trained = state() if callable(state) else None
        return run_dirgcn(x, ei, pr_pairs,
                          attention=getattr(self.dirgcn, "attention", False),
                          trained_state=trained)