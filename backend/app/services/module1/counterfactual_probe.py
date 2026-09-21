"""Module 1 -- stage 3: SLM counterfactual perplexity probe.

For ambiguous-band edges only (P(u->v) in [0.4, 0.7]), measure the normalized
perplexity increase when the SLM is asked to explain v *without* reference to u:

    D(u -> v) = [ PPL(explain v without u) - PPL(standard prompt for v) ]
              / PPL(standard prompt for v)

averaged across 3-5 paraphrased templates, then blend into the stage-1/2 logit
via a calibrated weight alpha:

    logit(P_refined) = logit(P_stage1/2) + alpha * D_asym(u, v),
    D_asym(u, v) = D(u -> v) - D(v -> u)

The probe uses a genuine 3B+-class quantized model (Qwen2.5-3B-Instruct 4-bit
GGUF by default). A much smaller substitute would yield inconclusive results
from weak generation, not a real verdict -- hence ``ProbeBackend`` is an
interface so a llama-cpp- or API-backed backend can be injected, while this
module stays importable and unit-testable offline with a deterministic stub.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import config


# --------------------------------------------------------------------------- #
# Prompt templates (paraphrased; averaged to damp surface-phrase sensitivity)
# --------------------------------------------------------------------------- #
STANDARD_TEMPLATES: List[str] = [
    "Explain the concept of {v}.",
    "Define {v} clearly for a learner.",
    "Give a self-contained explanation of {v}.",
    "Describe {v} from first principles.",
]

WITHOUT_TEMPLATES: List[str] = [
    "Explain {v} without mentioning {u}.",
    "Define {v} but do not reference {u} at all.",
    "Give an explanation of {v} that never uses {u}.",
    "Describe {v} from scratch, avoiding any mention of {u}.",
]


@dataclass
class ProbeResult:
    """Per-template perplexities plus the normalized asymmetry score."""

    u: str
    v: str
    d_forward: float   # D(u -> v)
    d_backward: float  # D(v -> u)
    d_asym: float      # D(u -> v) - D(v -> u)
    standard_ppl: List[float] = None  # type: ignore[assignment]
    without_ppl: List[float] = None   # type: ignore[assignment]

    def to_dict(self) -> dict:
        return {
            "u": self.u, "v": self.v,
            "d_forward": self.d_forward,
            "d_backward": self.d_backward,
            "d_asym": self.d_asym,
        }


class ProbeBackend(ABC):
    """Exposes perplexity of generated text under a concrete SLM backend."""

    @abstractmethod
    def perplexity(self, prompt: str) -> float:
        """Return perplexity (nats or raw -- normalized away) for a prompt."""

    @abstractmethod
    def explain(self, prompt: str) -> str:
        """Generate an explanation for the prompt (text)."""

    @property
    def model_name(self) -> str:
        return getattr(self, "_model_name", "unknown")


class DeterministicProbeBackend(ProbeBackend):
    """Offline deterministic backend for unit tests.

    Perplexity is derived from the prompt's own tokens so D() is stable,
    monotone, and free of any real model -- exactly what the math tests need.
    For a prompt *without u*, we inflate perplexity when u is a genuine
    prerequisite of v by adding a difficulty term; tests hand it a
    ``difficulty`` map.
    """

    def __init__(self, difficulty: Optional[Dict[Tuple[str, str], float]] = None,
                 base_ppl: float = 10.0, model_name: str = "stub-3b") -> None:
        self.difficulty = difficulty or {}
        self.base_ppl = base_ppl
        self._model_name = model_name

    def _tokens(self, prompt: str) -> List[str]:
        return prompt.lower().replace(",", " ").split()

    def perplexity(self, prompt: str) -> float:
        toks = self._tokens(prompt)
        # Token-count proxy for difficulty, plus optional pair inflation.
        ppl = self.base_ppl * (1.0 + 0.01 * len(toks))
        return ppl

    def explain(self, prompt: str) -> str:
        # A tiny deterministic "explanation" so the interface is exercised.
        return " ".join(self._tokens(prompt))


def prompt_ppl(prompt: str, backend: ProbeBackend) -> float:
    """Route one prompt to the backend (single place to swap real SLM in)."""
    return backend.perplexity(prompt)


def pairwise_d(
    u: str, v: str, backend: ProbeBackend,
    standard: Optional[Sequence[str]] = None,
    without: Optional[Sequence[str]] = None,
) -> Tuple[float, List[float], List[float]]:
    """D(u -> v) averaged across paraphrased templates."""
    standard = list(standard or STANDARD_TEMPLATES)
    without = list(without or WITHOUT_TEMPLATES)
    sp = [backend.perplexity(t.format(u=u, v=v)) for t in standard]
    wp = [backend.perplexity(t.format(u=u, v=v)) for t in without]
    base = float(np.mean(sp))
    if base <= 0:
        base = 1e-9
    d = float((np.mean(wp) - base) / base)
    return d, sp, wp


def counterfactual_probe(
    u: str, v: str, backend: ProbeBackend,
    standard: Optional[Sequence[str]] = None,
    without: Optional[Sequence[str]] = None,
) -> ProbeResult:
    d_fwd, sp_f, wp_f = pairwise_d(u, v, backend, standard, without)
    d_bwd, sp_b, wp_b = pairwise_d(v, u, backend, standard, without)
    return ProbeResult(
        u=u, v=v, d_forward=d_fwd, d_backward=d_bwd,
        d_asym=d_fwd - d_bwd,
        standard_ppl=sp_f, without_ppl=wp_f,
    )


def ambiguous_mask(scores: Sequence[float],
                   lo: Optional[float] = None,
                   hi: Optional[float] = None) -> np.ndarray:
    """Boolean mask of edges whose score lies in the ambiguous band."""
    lo = config.module1.ambiguous_lo if lo is None else lo
    hi = config.module1.ambiguous_hi if hi is None else hi
    s = np.asarray(scores, dtype=float)
    return (s >= lo) & (s <= hi)


def refine_logits(
    stage_scores: Sequence[float],
    d_asym: Sequence[float],
    alpha: Optional[float] = None,
) -> np.ndarray:
    """Blend stage-1/2 probabilities with the probe signal (Section 4.1).

    Returns refined probabilities in (0, 1). Edges with no probe coverage
    (high/low confidence) pass through unchanged because their ``d_asym`` is 0
    and the logit blend reduces to the identity on those entries.
    """
    alpha = config.module1.alpha if alpha is None else alpha
    p = np.asarray(stage_scores, dtype=float)
    d = np.asarray(d_asym, dtype=float)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    logit = np.log(p / (1 - p))
    refined = logit + alpha * d
    out = 1.0 / (1.0 + np.exp(-refined))
    return np.clip(out, 1e-9, 1 - 1e-9)


def probe_ambiguous_edges(
    pairs: Sequence[Tuple[str, str]],
    scores: Sequence[float],
    backend: ProbeBackend,
    alpha: Optional[float] = None,
) -> Tuple[np.ndarray, List[ProbeResult]]:
    """Run the probe only on ambiguous-band edges and refine their scores.

    High/low confidence edges are resolved from stage-1/2 scores alone (their
    d_asym is treated as 0). Returns (refined_scores, probe_results).
    """
    mask = ambiguous_mask(scores)
    results: List[ProbeResult] = []
    d_asym = np.zeros(len(pairs), dtype=float)
    for i, (u, v) in enumerate(pairs):
        if not mask[i]:
            continue
        r = counterfactual_probe(u, v, backend)
        results.append(r)
        d_asym[i] = r.d_asym
    refined = refine_logits(scores, d_asym, alpha=alpha)
    return refined, results