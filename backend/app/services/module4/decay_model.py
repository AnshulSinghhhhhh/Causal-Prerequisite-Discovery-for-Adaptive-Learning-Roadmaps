"""Module 4 -- memory-decay / spaced-repetition stability model.

Section 4.4: retention for concept i follows an exponential forgetting curve:

    R_i(dt) = exp( -dt / S_i )

where ``S_i`` is a per-concept, per-learner stability parameter. Rather than
inventing new decay mathematics, ``S_i`` is initialized and updated using an
established scheduler -- SM-2 or FSRS (Section 4.4, "SM-2 or FSRS, don't invent
new decay math"). Cold start uses a global default prior (day one, before any
quiz data): the SM-2 initial ease factor or the FSRS default stability, seeded
from aggregate first-attempt accuracy once usage data exists.

When ``R_i(dt) < tau_decay`` the corresponding edge is marked
``refresher_needed`` (see ``graph_rewriter.refresh_edge``).

Two scheduler backends live behind one ``StabilityScheduler`` interface so the
decay math (``retention`` / ``time_until_decay``) is identical either way and
unit-tested with synthetic curves.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from ..config import config


def retention(elapsed_seconds: float, stability: float) -> float:
    """R_i(dt) = exp(-dt / S_i). ``stability`` and ``elapsed`` share units."""
    if stability <= 0:
        raise ValueError("stability must be positive")
    if elapsed_seconds < 0:
        raise ValueError("elapsed time cannot be negative")
    return math.exp(-elapsed_seconds / stability)


def time_until_decay(stability: float, tau_decay: Optional[float] = None) -> float:
    """Solve R(dt) = tau_decay for dt => dt = S_i * ln(1/tau_decay)."""
    tau = config.module4.tau_decay if tau_decay is None else tau_decay
    if tau <= 0 or tau >= 1:
        raise ValueError("tau_decay must be in (0, 1)")
    return stability * math.log(1.0 / tau)


class StabilityScheduler(ABC):
    """Updates stability S_i after a review; implements SM-2 or FSRS."""

    @abstractmethod
    def update(self, stability: float, quality: int,
               previous: Optional[float] = None,
               repetitions: int = 0,
               interval_days: float = 0.0) -> float:
        """Return the updated stability given review quality (0-5)."""


class SM2Scheduler(StabilityScheduler):
    """SM-2: ease-factor (EF) update mapped onto a stability estimate.

    SM-2's EF and interval are converted to a stability value so the forgetting
    curve ``R(dt)`` stays the single canonical retention representation. The
    ease factor is the canonical SM-2 quantity; ``stability`` here is expressed
    in the same seconds unit the rest of Module 4 uses (callers pass seconds and
    receive seconds back).
    """

    ease_initial: float = config.module4.ease_initial

    def update(self, stability: float, quality: int,
               previous: Optional[float] = None,
               repetitions: int = 0,
               interval_days: float = 0.0) -> float:
        q = float(max(0, min(5, quality)))
        ef = previous if previous is not None else self.ease_initial
        # Canonical SM-2 EF update.
        ef = ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
        ef = max(1.3, ef)
        # SM-2 interval progression; convert to a stability (seconds).
        if q < 3:
            rep = 0
            ivl_days = 1.0
        else:
            rep = repetitions + 1
            if rep == 1:
                ivl_days = 1.0
            elif rep == 2:
                ivl_days = 6.0
            else:
                ivl_days = interval_days * ef
        return ivl_days * 86400.0 * max(ef / self.ease_initial, 0.5)


class FSRSScheduler(StabilityScheduler):
    """FSRS-style stability update (four-parameter DSR model, simplified).

    Stability after a successful review grows multiplicatively; after a failure
    it shrinks. This is a faithful simplification of FSRS's stability recursion
    (no invented decay -- still the same exponential forgetting dynamic), kept
    free of external package dependencies so it runs offline.
    """

    default_stability: float = config.module4.fsrs_default_stability_days * 86400.0
    # Factors: [w0..w3] = (initial-stability multiplier, failure divisor,
    #                      success growth base, easy-bonus gain).
    # Failure DIVIDES stability by w[1] (must be > 1 to shrink); success
    # MULTIPLIES by a growth term >= 1.
    w = [0.4, 1.8, 1.6, 0.12]

    def update(self, stability: float, quality: int,
               previous: Optional[float] = None,
               repetitions: int = 0,
               interval_days: float = 0.0) -> float:
        if stability <= 0:
            stability = self.default_stability
        if quality >= 3:
            # Success: grow multiplicatively, more for easier reviews.
            growth = (self.w[2] + self.w[3] * max(0, quality - 3))
            return stability * growth
        # Failure: shrink multiplicatively by the failure divisor.
        return max(stability / self.w[1], 60.0)  # floor ~1 min


class DecayModel:
    """Tracks retention per concept and flags decay-crossing edges.

    Backed by a ``StabilityScheduler`` and a ``GraphModel`` (for last-review
    timestamps and stability storage), so decay state lives on the same live
    graph object the service holds -- no recomputed copy.
    """

    def __init__(
        self,
        graph,
        scheduler: Optional[StabilityScheduler] = None,
        now_fn=__import__("time").time,
    ) -> None:
        self.graph = graph
        self.scheduler = scheduler or self._default_scheduler()
        self._now = now_fn

    @staticmethod
    def _default_scheduler() -> StabilityScheduler:
        if config.module4.scheduler == "fsrs":
            return FSRSScheduler()
        return SM2Scheduler()

    def initialize_stability(self, node_id: str,
                             prior: Optional[float] = None) -> float:
        """Set S_i using a difficulty prior or the global cold-start default."""
        if prior is not None:
            self.graph.set_stability(node_id, prior)
            return prior
        default = (FSRSScheduler.default_stability
                   if isinstance(self.scheduler, FSRSScheduler)
                   else 86400.0 * 1.0)  # 1 day cold start for SM-2
        self.graph.set_stability(node_id, default)
        return default

    def record_review(self, node_id: str, quality: int,
                      timestamp: Optional[float] = None) -> float:
        """Update stability after a review and record the timestamp."""
        ts = timestamp if timestamp is not None else self._now()
        current = self.graph.get_stability(node_id) \
            or self.initialize_stability(node_id)
        new_s = self.scheduler.update(current, quality)
        self.graph.set_stability(node_id, new_s)
        self.graph.set_last_review(node_id, ts)
        return new_s

    def retention(self, node_id: str, now: Optional[float] = None) -> float:
        now = now if now is not None else self._now()
        last = self.graph.get_last_review(node_id)
        if last is None:
            # Never reviewed: cold-start retention against initialized S.
            self.initialize_stability(node_id)
            last = now
            self.graph.set_last_review(node_id, last)
        s = self.graph.get_stability(node_id) or self.initialize_stability(node_id)
        return retention(max(0.0, now - last), s)

    def is_decayed(self, node_id: str, now: Optional[float] = None) -> bool:
        tau = config.module4.tau_decay
        return self.retention(node_id, now=now) < tau


def difficulty_prior(first_attempt_accuracy: float) -> float:
    """Seed S_i from aggregate first-attempt accuracy (Section 4.4).

    Higher aggregate accuracy -> higher starting stability. Maps accuracy in
    [0, 1] to a stability in [0.5, 3] days, deterministic and monotone.
    """
    acc = max(0.0, min(1.0, first_attempt_accuracy))
    days = 0.5 + 2.5 * acc
    return days * 86400.0