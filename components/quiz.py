"""Quiz component (spec §48). Question kinds are all rendered as choice
questions but their *content* varies: definition, matching, diagram
interpretation, equation interpretation, code interpretation, output
prediction, shape calculation, curve diagnosis, error diagnosis, scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st


@dataclass
class Q:
    text: str                       # Arabic question (Markdown allowed)
    options: list[str]
    answer: int                     # index into options
    explain: str = ""               # shown after submission
    code: str | None = None         # optional code block shown with the question
    latex: str | None = None        # optional equation shown with the question
    kind: str = "definition"        # for authoring clarity / tests


def quiz(key: str, questions: list[Q], title_ar: str = "اختبر فهمك") -> None:
    store = st.session_state.quiz.setdefault(key, {"submitted": False, "answers": {}})
    st.markdown(f"### 📝 {title_ar}")
    with st.form(key=f"quiz_form_{key}", border=True):
        for i, q in enumerate(questions):
            st.markdown(f"**{i + 1}. {q.text}**")
            if q.code:
                st.code(q.code, language="python")
            if q.latex:
                st.latex(q.latex)
            st.radio(
                "اختر إجابة",
                options=list(range(len(q.options))),
                format_func=lambda j, opts=q.options: opts[j],
                key=f"quiz_{key}_{i}",
                index=None,
                label_visibility="collapsed",
            )
        submitted = st.form_submit_button("تحقق من الإجابات", type="primary", icon=":material/check:")
    if submitted:
        store["submitted"] = True
        store["answers"] = {i: st.session_state.get(f"quiz_{key}_{i}") for i in range(len(questions))}
    if store["submitted"]:
        answers = store["answers"]
        correct = sum(1 for i, q in enumerate(questions) if answers.get(i) == q.answer)
        st.markdown(f"**النتيجة: {correct} / {len(questions)}**")
        for i, q in enumerate(questions):
            given = answers.get(i)
            if given is None:
                st.warning(f"السؤال {i + 1}: لم تُجب.", icon="⚠️")
            elif given == q.answer:
                st.success(f"السؤال {i + 1}: إجابة صحيحة. {q.explain}", icon="✅")
            else:
                st.error(
                    f"السؤال {i + 1}: الإجابة الصحيحة هي «{q.options[q.answer]}». {q.explain}",
                    icon="❌",
                )
        if st.button("إعادة المحاولة", key=f"quiz_reset_{key}", type="tertiary"):
            store["submitted"] = False
            store["answers"] = {}
            for i in range(len(questions)):
                st.session_state.pop(f"quiz_{key}_{i}", None)
            st.rerun()
