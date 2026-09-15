"""Researcher-note callouts (spec §53) and semantic callouts (spec §15).

Every callout has icon + label + colour — never colour alone.
"""

from __future__ import annotations

import streamlit as st

from core.rtl import esc

# kind -> (icon, Arabic label, English label, css class)
KINDS = {
    "research":   ("🔬", "ملاحظة بحثية", "Research Note", "sem-note"),
    "practical":  ("🛠️", "ملاحظة عملية", "Practical Note", "sem-code"),
    "interpret":  ("🧠", "ملاحظة تفسيرية", "Interpretation Note", "sem-math"),
    "mistake":    ("🚫", "خطأ شائع", "Common Mistake", "sem-fail"),
    "warning":    ("⚠️", "تحذير", "Warning", "sem-warn"),
    "math":       ("∑", "ملاحظة رياضية", "Math Note", "sem-math"),
    "coding":     ("💻", "ملاحظة برمجية", "Coding Note", "sem-code"),
    "debugging":  ("🐞", "ملاحظة تشخيصية", "Debugging Note", "sem-loss"),
    "intuition":  ("💡", "حدس", "Intuition", "sem-ok"),
    "why":        ("❓", "لماذا نحتاجه؟", "Why?", "sem-hyper"),
    "definition": ("📘", "تعريف", "Definition", "sem-data"),
    "takeaway":   ("🎯", "الخلاصة", "Key Takeaway", "sem-grad"),
    "data":       ("📥", "بيانات / مدخلات", "Input / Data", "sem-data"),
    "param":      ("🎛️", "معلمة", "Parameter", "sem-param"),
    "hyper":      ("🔧", "معلمة فائقة", "Hyperparameter", "sem-hyper"),
    "loss":       ("📉", "خسارة", "Loss", "sem-loss"),
    "grad":       ("🧭", "تدرج", "Gradient", "sem-grad"),
    "ok":         ("✅", "صحيح", "Correct", "sem-ok"),
    "fail":       ("❌", "فشل", "Failure", "sem-fail"),
}


def _paragraphs(body: str) -> str:
    """Very small Markdown subset for callout bodies: paragraphs, `code`, **bold**."""
    import re

    out = []
    for para in body.strip().split("\n\n"):
        p = esc(para.strip())
        p = re.sub(r"`([^`]+)`", r'<code class="ltr">\1</code>', p)
        p = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", p)
        p = p.replace("\n", "<br>")
        out.append(f"<p>{p}</p>")
    return "".join(out)


def callout(kind: str, body: str, title: str | None = None) -> None:
    icon, ar, en, cls = KINDS[kind]
    label = title or f"{ar} · {en}"
    st.html(
        f'<div class="dlia-callout {cls}">'
        f'<div class="dlia-callout-title"><span>{icon}</span><span>{esc(label)}</span></div>'
        f"{_paragraphs(body)}</div>"
    )


# Convenience wrappers — one per note type from spec §53
def research_note(body: str, title: str | None = None) -> None: callout("research", body, title)
def practical_note(body: str, title: str | None = None) -> None: callout("practical", body, title)
def interpretation_note(body: str, title: str | None = None) -> None: callout("interpret", body, title)
def common_mistake(body: str, title: str | None = None) -> None: callout("mistake", body, title)
def warning_note(body: str, title: str | None = None) -> None: callout("warning", body, title)
def math_note(body: str, title: str | None = None) -> None: callout("math", body, title)
def coding_note(body: str, title: str | None = None) -> None: callout("coding", body, title)
def debugging_note(body: str, title: str | None = None) -> None: callout("debugging", body, title)
def intuition(body: str, title: str | None = None) -> None: callout("intuition", body, title)
def why(body: str, title: str | None = None) -> None: callout("why", body, title)
def definition(body: str, title: str | None = None) -> None: callout("definition", body, title)
def takeaway(body: str, title: str | None = None) -> None: callout("takeaway", body, title)
