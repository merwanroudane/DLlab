"""Shared layout for official-course week pages (spec §29): the overview page
(this-week pipeline, foundations links, pre-test) and the post-test block.
Every week keeps the same rhythm so learners always know where they are."""

from __future__ import annotations

from typing import Sequence

import streamlit as st

from components.callouts import practical_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline, table


def week_overview(lesson: Lesson, *, steps: Sequence[str], links: Sequence[tuple[str, str, str]], buttons: Sequence[tuple[str, str, str]],
                  pretest: Sequence[Q], note_ar: str, takeaway_ar: str, week_no: str) -> None:
    """links: (topic, week lesson, foundations lesson). buttons: (label, icon, route)."""
    lesson_header(lesson)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(list(steps), active=0)
    table(["الموضوع", "درس الأسبوع", "الأسس التي يعتمد عليها"], list(links), ["rtl", "rtl", "rtl"])
    practical_note(note_ar)
    with st.container(horizontal=True):
        for i, (label, icon, route) in enumerate(buttons):
            st.button(label, icon=icon, on_click=go, args=(route,), key=f"{lesson.id}_btn_{i}")
    h2("الاختبار القبلي", "Pre-test")
    st.markdown("اختبار قصير **قبل** الدرس لقياس نقطة انطلاقك. لا يُحتسب؛ استخدمه لتعرف ما تراجعه.")
    quiz(f"{lesson.module}.pretest", list(pretest), title_ar=f"الاختبار القبلي — الأسبوع {week_no}")
    takeaway(takeaway_ar)
    lesson_footer(lesson, ["خريطة الأسبوع وروابط الأسس.", "الاختبار القبلي."])


def post_test(module_id: str, week_no: str, questions: Sequence[Q]) -> None:
    h2("الاختبار البعدي", "Post-test")
    st.markdown("نفس روح الاختبار القبلي مع أسئلة تطبيقية إضافية؛ قارن نتيجتك بما قبل الأسبوع.")
    quiz(f"{module_id}.posttest", list(questions), title_ar=f"الاختبار البعدي — الأسبوع {week_no}")
