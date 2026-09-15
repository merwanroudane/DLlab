"""Shared scoring for the readiness module: quiz results and challenge
results are recorded in session state so `ready_status` can compute a
status with concrete gaps."""

from __future__ import annotations

import streamlit as st

from components.quiz import Q, quiz


def scores() -> dict:
    return st.session_state.setdefault("ready_scores", {})


def record(name: str, correct: int, total: int, module_id: str) -> None:
    scores()[name] = {"correct": int(correct), "total": int(total), "module": module_id}


def scored_quiz(name: str, key: str, questions: list[Q], module_id: str, title_ar: str = "اختبر فهمك") -> None:
    """Render a quiz and, once submitted, record its score under `name`."""
    quiz(key, questions, title_ar=title_ar)
    store = st.session_state.quiz.get(key, {})
    if store.get("submitted"):
        answers = store.get("answers", {})
        correct = sum(1 for i, q in enumerate(questions) if answers.get(i) == q.answer)
        record(name, correct, len(questions), module_id)


def check_answer(name: str, module_id: str, ok: bool, total: int = 1) -> None:
    """Record a single-item challenge (ok → 1/1)."""
    record(name, 1 if ok else 0, total, module_id)
