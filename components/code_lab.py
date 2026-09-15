"""Code Lab (spec §22, §22A, §23): every code example follows the Code
Understanding Contract — Before the code → Read the code (line by line) →
Run (real output) → After the code (interpretation).

Code is *predefined and safe*: the lab executes a Python callable shipped
with the lesson, driven by parameter controls — never `exec` on user text.
"""

from __future__ import annotations

import contextlib
import io
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import streamlit as st

from core.rtl import esc


def exec_source(code: str, params: dict[str, Any] | None = None) -> str:
    """Execute *authored* lesson code (never user input) and return its stdout.

    `params` are injected as globals so parameter controls can drive the code.
    Exceptions are returned as text so the learner sees the real traceback.
    """
    ns: dict[str, Any] = {"__name__": "__lesson__"}
    if params:
        ns.update(params)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<lesson>", "exec"), ns)  # noqa: S102 - trusted, authored code
    except Exception:  # noqa: BLE001 - we want to show the real error to the learner
        buf.write("\n" + traceback.format_exc(limit=2))
    return buf.getvalue()


def fill_template(code: str, params: dict[str, Any]) -> str:
    """Fill `{name}` placeholders with control values.

    Tries `str.format` first (so authors may escape braces as `{{`); if the
    code contains other braces (f-strings, dict literals) that would break
    `format`, fall back to substituting only the known parameter names.
    """
    import re

    try:
        return code.format(**params)
    except (KeyError, IndexError, ValueError):
        return re.sub(r"\{(\w+)\}", lambda m: str(params[m.group(1)]) if m.group(1) in params else m.group(0), code)


def run_printed(code: str, template: bool = False) -> Callable[[dict[str, Any]], None]:
    """Build a `run` callable that executes `code` (formatted with params if
    `template`) and shows its printed output."""

    def _run(params: dict[str, Any]) -> None:
        src = fill_template(code, params) if template else code
        out = exec_source(src, params)
        st.code(out.rstrip() or "(لا مخرجات مطبوعة)", language="text")

    return _run


@dataclass
class Before:
    goal_ar: str
    stage_ar: str                          # where are we in the pipeline?
    inputs_ar: str
    expected_ar: str
    math_ar: str | None = None
    prerequisites_ar: str | None = None
    objects_ar: str | None = None          # framework objects we will create


@dataclass
class CodeLab:
    key: str
    title_ar: str
    code: str
    before: Before
    explain: list[tuple[str, str]]         # [("1-3", "arabic explanation"), ...]
    run: Callable[[dict[str, Any]], None]  # renders outputs with st.*
    controls: Callable[[], dict[str, Any]] | None = None
    after_ar: str | None = None            # how to read the output
    level: str = "A"                       # A concept | B framework | C pipeline
    language: str = "python"
    step_run: Callable[[dict[str, Any], int], int] | None = None
    """Optional step-by-step runner: (params, step_index) -> total_steps.
    Renders the state after `step_index` steps and returns the total count."""
    template: bool = False
    """If True, `code` is a str.format template filled with the control values
    (defaults when the controls are not shown), so the displayed code always
    matches what runs."""
    defaults: dict[str, Any] = field(default_factory=dict)


LEVEL_AR = {"A": "المستوى A — كود المفهوم", "B": "المستوى B — كود إطار العمل", "C": "المستوى C — خط أنابيب تطبيقي"}


def _line_map(code: str, explain: list[tuple[str, str]]) -> None:
    lines = code.rstrip("\n").split("\n")
    for rng, text in explain:
        if "-" in rng:
            a, b = (int(x) for x in rng.split("-"))
        else:
            a = b = int(rng)
        snippet = "\n".join(lines[a - 1:b])
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.code(snippet, language="python", line_numbers=False)
        with c2:
            st.markdown(f"**الأسطر {rng}:** {text}")


def code_lab(lab: CodeLab) -> None:
    st.markdown(f"### 💻 {lab.title_ar}")
    st.caption(LEVEL_AR.get(lab.level, lab.level))
    b = lab.before
    with st.expander("قبل الكود — ماذا سنفعل ولماذا؟", expanded=True, icon=":material/flag:"):
        st.markdown(f"**🎯 الهدف:** {b.goal_ar}")
        st.markdown(f"**📍 أين نحن في خط الأنابيب؟** {b.stage_ar}")
        if b.prerequisites_ar:
            st.markdown(f"**📚 المتطلبات:** {b.prerequisites_ar}")
        st.markdown(f"**📥 المدخلات:** {b.inputs_ar}")
        st.markdown(f"**📤 المخرجات المتوقعة:** {b.expected_ar}")
        if b.math_ar:
            st.markdown(f"**∑ المقابل الرياضي:** {b.math_ar}")
        if b.objects_ar:
            st.markdown(f"**🧱 الكائنات التي سننشئها:** {b.objects_ar}")

    shown_code = fill_template(lab.code, lab.defaults) if lab.template and lab.defaults else lab.code
    view = st.segmented_control(
        "العرض",
        options=["الكود", "شرح سطرًا سطرًا", "تشغيل"] + (["تشغيل خطوة بخطوة"] if lab.step_run else []),
        default="الكود",
        key=f"codelab_view_{lab.key}",
        label_visibility="collapsed",
    )
    if os.environ.get("DLIA_FORCE_RUN_CODELABS"):
        view = "تشغيل"  # test hook: execute every code lab headlessly
    if view == "الكود":
        st.code(shown_code, language=lab.language, line_numbers=True)
    elif view == "شرح سطرًا سطرًا":
        _line_map(shown_code, lab.explain)
    elif view == "تشغيل":
        params = dict(lab.defaults)
        if lab.controls:
            params.update(lab.controls())
        if lab.template:
            st.code(fill_template(lab.code, params), language=lab.language)
        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("إعادة الضبط", key=f"codelab_reset_{lab.key}", icon=":material/restart_alt:"):
                for k in list(st.session_state.keys()):
                    if k.startswith(f"ctrl_{lab.key}_"):
                        del st.session_state[k]
                st.rerun()
        with st.container(border=True):
            st.markdown("**📤 المخرجات**")
            lab.run(params)
        if lab.after_ar:
            with st.expander("بعد الكود — كيف أقرأ هذه المخرجات؟", expanded=True, icon=":material/visibility:"):
                st.markdown(lab.after_ar)
    elif view == "تشغيل خطوة بخطوة" and lab.step_run:
        params = lab.controls() if lab.controls else {}
        step_key = f"codelab_step_{lab.key}"
        st.session_state.setdefault(step_key, 0)
        with st.container(horizontal=True):
            if st.button("السابق", key=f"{step_key}_prev", icon=":material/arrow_forward:"):
                st.session_state[step_key] = max(0, st.session_state[step_key] - 1)
            if st.button("التالي", key=f"{step_key}_next", icon=":material/arrow_back:", type="primary"):
                st.session_state[step_key] += 1
            if st.button("من البداية", key=f"{step_key}_restart", icon=":material/restart_alt:"):
                st.session_state[step_key] = 0
        with st.container(border=True):
            total = lab.step_run(params, st.session_state[step_key])
            if st.session_state[step_key] >= total:
                st.session_state[step_key] = total
            st.caption(f"الخطوة {min(st.session_state[step_key], total)} من {total}")
