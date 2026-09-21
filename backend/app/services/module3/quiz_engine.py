"""Module 3 -- diagnostic quiz engine with misconception-tagged distractors.

Section 4.3: each concept node is attached to exactly three multiple-choice
questions whose distractors are *explicitly pre-mapped* to a root misconception
vector ``M_k``:

* **Option A (correct)** -- confirms mastery of the target concept.
* **Option B (prerequisite distractor)** -- flags failure in an upstream
  dependency (e.g. a matrix-dimension mismatch).
* **Option C (conceptual distractor)** -- flags confusion within the current
  concept (e.g. correlation vs. causation).

A learner's incorrect selection is attributed to misconception ``M_k`` when the
match score falls below a threshold (paper default 0.1, grid-searched jointly
with Module 2's ``tau_edge`` against the same held-out validation portion --
Section 4.2.1). Selecting Option B for a target node ``v`` carries ``(target
node, misconception id)`` so Module 4 can act without a second lookup (Section 7
of the brief).

Question generation itself is deliberately injectable: the offline environment
uses a deterministic generator, while a real deployment swaps in an SLM-backed
generator that produces the same ``Question`` structure.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import config
from ..graph_model import EdgeType
from .content_aggregator import _cosine  # canonical cosine, one implementation


class OptionRole(str, Enum):
    CORRECT = "A"
    PREREQUISITE_DISTRACTOR = "B"
    CONCEPTUAL_DISTRACTOR = "C"


@dataclass
class Option:
    text: str
    role: OptionRole
    #: Misconception id (``M_k``) this distractor is pre-mapped to.
    misconception_id: Optional[str] = None
    #: Explanatory vector for the misconception (used for match scoring).
    misconception_vector: Optional[np.ndarray] = None


@dataclass
class Question:
    stem: str
    options: List[Option]
    #: Node this question targets (the concept being quizzed).
    target_node: str

    def correct_index(self) -> int:
        for i, o in enumerate(self.options):
            if o.role == OptionRole.CORRECT:
                return i
        raise ValueError("question has no correct option")


@dataclass
class QuizResult:
    """Outcome of a learner selecting an option on a question."""

    question: Question
    selected_index: int
    is_correct: bool
    attributed_misconception: Optional[str] = None
    match_score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "target_node": self.question.target_node,
            "selected_index": self.selected_index,
            "is_correct": self.is_correct,
            "attributed_misconception": self.attributed_misconception,
            "match_score": self.match_score,
        }


@dataclass
class Misconception:
    """A root misconception vector ``M_k`` and its metadata."""

    id: str
    description: str
    vector: np.ndarray
    kind: str = "prerequisite"  # or "conceptual"

    @classmethod
    def from_text(cls, id: str, description: str, dim: int = 384):
        # Deterministic embedding for offline use (mirrors _bow_embedding but
        # keyed to the misconception text so it is stable across runs).
        from .content_aggregator import _bow_embedding

        return cls(id=id, description=description,
                   vector=_bow_embedding(description, dim=dim))


def match_score(
    misconception: Misconception, option_vector: np.ndarray
) -> float:
    """Cosine similarity between a distractor's vector and misconception M_k.

    The paper's threshold is applied to this score; a *low* score attributes the
    selection to ``M_k`` (distractor is dissimilar from correct + similar to the
    misconception's negative space -- see ``attribute`` below for the exact
    decision rule the paper specifies).
    """
    return _cosine(misconception.vector, option_vector)


def attribute_selection(
    option: Option,
    misconception: Misconception,
    threshold: Optional[float] = None,
) -> Tuple[bool, float]:
    """Decide whether selecting ``option`` is attributed to ``M_k``.

    Paper default: attributed when match score < 0.1. Returns (attributed,
    score).
    """
    thresh = config.module3.gamma_misconception if threshold is None else threshold
    if option.misconception_vector is None:
        return False, float("nan")
    score = match_score(misconception, option.misconception_vector)
    return score < thresh, score


class QuizEngine:
    """Generates quizzes and attributes selections to misconceptions."""

    def __init__(
        self,
        misconception_bank: Optional[Sequence[Misconception]] = None,
        generator=None,
    ) -> None:
        self.bank: Dict[str, Misconception] = {
            m.id: m for m in (misconception_bank or [])
        }
        self.generator = generator or DeterministicQuestionGenerator()

    def add_misconception(self, m: Misconception) -> None:
        self.bank[m.id] = m

    def generate(self, node_id: str, concept_text: str,
                 prerequisites: Sequence[str] = (),
                 misconceptions: Sequence[str] = ()) -> List[Question]:
        """Attach exactly three questions per node (questions_per_node)."""
        qs = self.generator.generate(
            node_id=node_id, concept_text=concept_text,
            prerequisites=prerequisites, misconceptions=misconceptions,
            n=config.module3.questions_per_node,
        )
        # Enforce exactly three options per question with the canonical roles.
        for q in qs:
            self._validate_question(q)
        return qs

    def grade(self, question: Question, selected_index: int,
              misconception_id: Optional[str] = None) -> QuizResult:
        option = question.options[selected_index]
        is_correct = option.role == OptionRole.CORRECT
        result = QuizResult(
            question=question, selected_index=selected_index,
            is_correct=is_correct,
        )
        if is_correct:
            return result

        # A distractor is PRE-MAPPED to a misconception (Section 4.3): the
        # option's own id is the canonical attribution. When the bank holds a
        # full vector for that id, the match-score threshold refines the call;
        # otherwise the pre-map id is used directly (no bank needed).
        mapped_id = option.misconception_id or misconception_id
        if mapped_id is not None:
            m = self.bank.get(mapped_id)
            if m is not None and option.misconception_vector is not None:
                attributed, score = attribute_selection(option, m)
                result.match_score = score
                result.attributed_misconception = m.id if attributed else None
            else:
                # Direct pre-map (no vector refinment available).
                result.attributed_misconception = mapped_id
        return result

    @staticmethod
    def _validate_question(q: Question) -> None:
        roles = [o.role for o in q.options]
        if roles.count(OptionRole.CORRECT) != 1:
            raise ValueError("each quiz question needs exactly one correct option")
        if roles.count(OptionRole.PREREQUISITE_DISTRACTOR) != 1:
            raise ValueError("each quiz question needs exactly one Option B")
        if roles.count(OptionRole.CONCEPTUAL_DISTRACTOR) != 1:
            raise ValueError("each quiz question needs exactly one Option C")


class DeterministicQuestionGenerator:
    """Offline generator producing well-formed three-option questions.

    A real deployment swaps in an SLM generator producing the same structure;
    the deterministic generator keeps Module 3 (and Module 4, which consumes its
    outcomes) fully unit-testable without a model or network.
    """

    def generate(self, node_id: str, concept_text: str,
                 prerequisites: Sequence[str] = (),
                 misconceptions: Sequence[str] = (),
                 n: int = 3) -> List[Question]:
        qs: List[Question] = []
        # A stable "topic vector" for the concept, used to build misconception
        # vectors whose cosine-distance drives the match threshold.
        from .content_aggregator import _bow_embedding

        topic = _bow_embedding(concept_text, dim=384)
        for i in range(n):
            stem = f"Which statement about {concept_text} is correct? ({i + 1})"
            # Correct option vector = topic (max similarity to concepts).
            opt_a = Option(
                text=f"The definition of {concept_text}.",
                role=OptionRole.CORRECT,
                misconception_vector=topic.copy(),
            )
            # Prerequisite distractor: wrong but tied to an upstream dep.
            prereq = prerequisites[0] if prerequisites else "the prerequisite"
            mc_prereq_id = (misconceptions[0] if misconceptions
                            else f"M_prereq_{node_id}")
            opt_b = Option(
                text=f"A statement confusing {concept_text} with {prereq}.",
                role=OptionRole.PREREQUISITE_DISTRACTOR,
                misconception_id=self._canonical_id(mc_prereq_id),
                misconception_vector=topic * -0.8,  # dissimilar -> low match
            )
            # Conceptual distractor: confusion within the current concept.
            mc_conc_id = (misconceptions[1] if len(misconceptions) > 1
                          else f"M_concept_{node_id}")
            opt_c = Option(
                text=f"A statement reversing the direction of {concept_text}.",
                role=OptionRole.CONCEPTUAL_DISTRACTOR,
                misconception_id=self._canonical_id(mc_conc_id),
                misconception_vector=topic * 0.05,  # near-zero similarity
            )
            qs.append(Question(stem=stem,
                               options=[opt_a, opt_b, opt_c],
                               target_node=node_id))
        return qs

    @staticmethod
    def _canonical_id(m: str) -> str:
        return f"M_{hashlib.md5(m.encode('utf-8')).hexdigest()[:8]}"


def build_misconception_bank(
    entries: Sequence[Tuple[str, str, str]],
    dim: int = 384,
) -> Dict[str, Misconception]:
    """Build a misconception bank from (id, description, kind) triples."""
    bank = {}
    for mid, desc, kind in entries:
        bank[mid] = Misconception.from_text(mid, desc, dim=dim)
        bank[mid].kind = kind
    return bank