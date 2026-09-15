"""Problem / diagnosis card — the 18-point framework of spec §33, rendered
inside the lesson where the problem can occur (spec §32)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import streamlit as st

from components.comparison import good_vs_bad
from components.quiz import Q, quiz


@dataclass
class Problem:
    key: str
    name_ar: str
    name_en: str
    description_ar: str
    symptoms_ar: list[str]
    sees_ar: list[str]                       # what the researcher sees (logs, curves, error text)
    possible_causes_ar: list[str]
    root_causes_ar: list[str]
    diagnosis_ar: list[str]                  # ordered workflow
    evidence_ar: list[str]                   # what confirms it
    fixes_ar: list[str]
    tradeoffs_ar: list[str] = field(default_factory=list)
    misdiagnosis_ar: list[str] = field(default_factory=list)
    related_ar: list[str] = field(default_factory=list)
    checklist_ar: list[str] = field(default_factory=list)
    math_ar: str | None = None               # mathematical explanation (Markdown + LaTeX)
    error_text: str | None = None            # a realistic error / log excerpt
    wrong_code: str | None = None
    correct_code: str | None = None
    simulation: Callable[[], None] | None = None     # interactive simulation renderer
    experiment: Callable[[], None] | None = None     # code experiment renderer
    challenge: list[Q] = field(default_factory=list) # diagnostic challenge questions


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


def problem_card(p: Problem) -> None:
    with st.container(border=True):
        st.markdown(f"### 🩺 مشكلة: {p.name_ar} <span class='en' style='font-size:.8em'>{p.name_en}</span>",
                    unsafe_allow_html=True)
        st.markdown(p.description_ar)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🔎 الأعراض**")
            st.markdown(_bullets(p.symptoms_ar))
            st.markdown("**👀 ما يراه الباحث**")
            st.markdown(_bullets(p.sees_ar))
            if p.error_text:
                st.code(p.error_text, language="text")
        with c2:
            st.markdown("**❓ الأسباب المحتملة**")
            st.markdown(_bullets(p.possible_causes_ar))
            st.markdown("**🌱 الأسباب الجذرية**")
            st.markdown(_bullets(p.root_causes_ar))

        st.markdown("**🧭 خطوات التشخيص (بالترتيب)**")
        st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(p.diagnosis_ar, start=1)))
        st.markdown("**📎 الدليل الذي يؤكد التشخيص**")
        st.markdown(_bullets(p.evidence_ar))

        if p.simulation:
            with st.expander("🎛️ محاكاة تفاعلية", expanded=False):
                p.simulation()
        if p.math_ar:
            with st.expander("∑ التفسير الرياضي", expanded=False):
                st.markdown(p.math_ar)
        if p.experiment:
            with st.expander("💻 تجربة برمجية", expanded=False):
                p.experiment()
        if p.wrong_code or p.correct_code:
            good_vs_bad("الكود الصحيح", "", "الكود الخطأ", "",
                        good_code=p.correct_code, bad_code=p.wrong_code)

        st.markdown("**🛠️ الإصلاحات**")
        st.markdown(_bullets(p.fixes_ar))
        if p.tradeoffs_ar:
            st.markdown("**⚖️ المقايضات**")
            st.markdown(_bullets(p.tradeoffs_ar))
        if p.misdiagnosis_ar:
            st.markdown("**🚫 تشخيص خاطئ شائع**")
            st.markdown(_bullets(p.misdiagnosis_ar))
        if p.related_ar:
            st.markdown("**🔗 مشكلات مرتبطة**")
            st.markdown(_bullets(p.related_ar))
        if p.checklist_ar:
            st.markdown("**✅ قائمة تحقق**")
            for i, item in enumerate(p.checklist_ar):
                st.checkbox(item, key=f"chk_{p.key}_{i}")
        if p.challenge:
            quiz(f"diag_{p.key}", p.challenge, title_ar="تحدٍّ تشخيصي")
