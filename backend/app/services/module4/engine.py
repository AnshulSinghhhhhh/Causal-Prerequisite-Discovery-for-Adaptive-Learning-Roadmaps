"""Module 4 -- remediation + decay orchestration.

Binds the decay model and graph rewriter to the live ``GraphModel`` so quiz
outcomes and elapsed time mutate the graph in one place. This is the seam the
remediation router and the Track B simulation import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import config
from ..graph_model import GraphModel, NodeStatus
from ..module3.quiz_engine import QuizResult, OptionRole
from .decay_model import DecayModel, SM2Scheduler, FSRSScheduler
from .graph_rewriter import GraphRewriter


@dataclass
class Module4Config:
    pass


class RemediationEngine:
    """Consumes quiz outcomes + elapsed time to rewrite the live graph.

    A single instance is held per learner session. It owns a ``GraphRewriter``
    and a ``DecayModel`` over the same graph object, so decay flagging and
    misconception injection never diverge across two copied graphs.
    """

    def __init__(self, graph: GraphModel,
                 scheduler: Optional[str] = None) -> None:
        self.graph = graph
        self.decay = DecayModel(graph)
        self.rewriter = GraphRewriter(graph)

    # -- quiz-outcome handling ----------------------------------------- #
    def handle_quiz_result(self, result: QuizResult) -> dict:
        """Route a graded selection into remediation or decay updates.

        * Option B (prerequisite distractor) attributed to ``M_k`` -> inject a
          remediation node upstream of the target.
        * Any reviewed concept -> update its stability and review timestamp.
        Returns a dict suitable for the API response.
        """
        outcome: dict = {"action": "none"}
        target = result.question.target_node

        if result.attributed_misconception is not None:
            # Option B mapped to a prerequisite misconception triggers rewrite.
            rm = self.rewriter.inject_remediation(
                target, result.attributed_misconception
            )
            outcome = {"action": "remediate", **rm.to_dict()}
        elif result.is_correct:
            # A correct answer is a successful review -> bump stability.
            quality = config.module4.success_score
            self.decay.record_review(target, quality)
            self.graph.get_node(target).status = NodeStatus.COMPLETED
            outcome = {"action": "complete", "target_node": target}
        else:
            quality = max(1, config.module4.success_score - 2)
            self.decay.record_review(target, quality)
            outcome = {"action": "review", "target_node": target}

        return outcome

    # -- decay flagging ------------------------------------------------- #
    def decayed_nodes(self, now: Optional[float] = None) -> list:
        """Return node ids whose retention has fallen below tau_decay."""
        return [nid for nid in self.graph.node_ids()
                if self.decay.is_decayed(nid, now=now)]

    def check_and_refresh(self, now: Optional[float] = None) -> list:
        """Mark decayed dependencies as needing refresh; return changed edges."""
        changed = []
        for e in list(self.graph.edges()):
            if self.decay.is_decayed(e.target, now=now):
                self.rewriter.refresh_edge(e.source, e.target)
                changed.append((e.source, e.target))
        return changed