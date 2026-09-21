"""Module 3 quiz endpoints: generate quizzes per node and grade selections."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..models.schema import QuizQuestionPayload, QuizResponsePayload
from ..session_store import session_store
from ..services.module3.quiz_engine import (
    DeterministicQuestionGenerator,
    OptionRole,
    QuizEngine,
)

from ..services.content_store import get_node_content, complete_and_unlock_next
from ..services.module3.quiz_engine import Question, Option, OptionRole, QuizResult

router = APIRouter(prefix="/quiz", tags=["quiz"])

#: Default quiz engine (bank can be extended via the remediation router).
_engine = QuizEngine(generator=DeterministicQuestionGenerator())


class QuizGenerateRequest(BaseModel):
    session_id: str
    node_id: str
    concept_text: Optional[str] = None
    prerequisites: List[str] = []


class QuizGradeRequest(BaseModel):
    session_id: str
    node_id: str
    stem: str
    selected_index: int


@router.post("/generate", response_model=List[QuizQuestionPayload])
def generate_quiz(req: QuizGenerateRequest) -> List[QuizQuestionPayload]:
    stored = get_node_content(req.session_id, req.node_id) or {}
    stored_quizzes = stored.get("quiz", [])
    if stored_quizzes:
        return [
            QuizQuestionPayload(
                stem=q["stem"],
                options=q["options"],
                correct_index=q.get("correct_index", 0),
                target_node=req.node_id,
            )
            for q in stored_quizzes
        ]

    concept_text = req.concept_text or req.node_id
    questions = _engine.generate(
        node_id=req.node_id, concept_text=concept_text,
        prerequisites=req.prerequisites,
    )
    return [
        QuizQuestionPayload(
            stem=q.stem,
            options=[o.text for o in q.options],
            correct_index=q.correct_index(),
            target_node=q.target_node,
        )
        for q in questions
    ]


@router.post("/grade", response_model=QuizResponsePayload)
def grade_quiz(req: QuizGradeRequest) -> QuizResponsePayload:
    """Grade a selection and route remediation when needed (via engine)."""
    try:
        engine = session_store.get_engine(req.session_id)
        graph = session_store.get_graph(req.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    # 1. Check stored dynamic questions first
    stored = get_node_content(req.session_id, req.node_id) or {}
    stored_quizzes = stored.get("quiz", [])
    for q_data in stored_quizzes:
        if q_data["stem"] == req.stem:
            correct_idx = q_data.get("correct_index", 0)
            is_correct = (req.selected_index == correct_idx)
            # Option B is prerequisite distractor (index 1)
            is_prereq_distractor = (req.selected_index == 1 and not is_correct)
            misc_tag = q_data.get("misconception_tag", f"M_{req.node_id}") if is_prereq_distractor else None

            # Form a QuizResult
            opts = [
                Option(text=t, role=OptionRole.CORRECT if i == correct_idx else (OptionRole.PREREQUISITE_DISTRACTOR if i == 1 else OptionRole.CONCEPTUAL_DISTRACTOR))
                for i, t in enumerate(q_data["options"])
            ]
            synth_q = Question(stem=q_data["stem"], options=opts, target_node=req.node_id)
            quiz_result = QuizResult(
                question=synth_q,
                selected_index=req.selected_index,
                is_correct=is_correct,
                attributed_misconception=misc_tag,
                match_score=0.05 if misc_tag else 0.95,
            )
            outcome = engine.handle_quiz_result(quiz_result)
            if is_correct:
                complete_and_unlock_next(graph, req.node_id)

            return QuizResponsePayload(
                target_node=req.node_id,
                selected_index=req.selected_index,
                is_correct=is_correct,
                attributed_misconception=misc_tag,
                match_score=quiz_result.match_score,
                action=outcome.get("action"),
            )

    # 2. Fallback to deterministic generator bank
    found = None
    for q in _engine.generate(node_id=req.node_id, concept_text=req.node_id):
        if q.stem == req.stem:
            found = q
            break
    if found is None:
        raise HTTPException(status_code=404, detail="question not found")
    if not (0 <= req.selected_index < len(found.options)):
        raise HTTPException(status_code=422, detail="selected_index out of range")

    result = _engine.grade(found, req.selected_index)
    outcome = engine.handle_quiz_result(result)
    if result.is_correct:
        complete_and_unlock_next(graph, req.node_id)

    return QuizResponsePayload(
        target_node=result.question.target_node,
        selected_index=result.selected_index,
        is_correct=result.is_correct,
        attributed_misconception=result.attributed_misconception,
        match_score=result.match_score,
        action=outcome.get("action"),
    )