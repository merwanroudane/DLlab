"""Bidirectional typography helpers (spec §9, §10).

Rules of thumb for content authors
----------------------------------
* Write Arabic prose as normal Markdown; the stylesheet flips it RTL.
* Put every English identifier, code token, path, shape tuple or variable
  in back-ticks: `batch_size = 32`, `X_train.shape`, `(batch, features)`.
  Back-ticks render as an isolated LTR run.
* Use `en("Learning Rate")` for a non-code English phrase you want LTR and
  isolated inside an Arabic sentence.
* Use `table(...)` for mixed-direction tables (each column has a direction).
* Equations go through `st.latex` or `$...$` — KaTeX is isolated LTR by CSS.
"""

from __future__ import annotations

import html as _html
from typing import Iterable, Literal, Sequence

import streamlit as st

ColDir = Literal["rtl", "ltr", "code", "num"]


def esc(text: str) -> str:
    return _html.escape(str(text), quote=True)


def inline(text: str) -> str:
    """Escape, then render light inline Markdown: `code` spans (LTR-isolated)
    and **bold**. Used for table cells so authors can mark identifiers."""
    import re

    t = esc(text)
    t = re.sub(r"`([^`]+)`", lambda m: '<code class="ltr">' + m.group(1) + "</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", lambda m: "<b>" + m.group(1) + "</b>", t)
    return t


def en(text: str) -> str:
    """English phrase, LTR-isolated, for use inside Arabic HTML/Markdown."""
    return f'<span class="en">{esc(text)}</span>'


def code(text: str) -> str:
    """Inline code token, LTR-isolated (HTML form of back-ticks)."""
    return f'<code class="ltr">{esc(text)}</code>'


def chip(text: str, kind: str) -> str:
    """Semantic chip: kind in data|param|hyper|loss|grad|ok|warn|fail."""
    return f'<span class="chip sem-{kind}">{esc(text)}</span>'


def rtl(html_body: str) -> str:
    return f'<div class="dlia-rtl">{html_body}</div>'


def md(text: str) -> None:
    """Arabic Markdown paragraph(s). Thin wrapper kept for symmetry/searchability."""
    st.markdown(text, unsafe_allow_html=True)


def html(body: str) -> None:
    st.html(body)


def table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    col_dirs: Sequence[ColDir] | None = None,
    *,
    caption: str | None = None,
    raw_html: bool = False,
) -> None:
    """Render a mixed-direction table.

    col_dirs: per-column direction — "rtl" (Arabic), "ltr" (English prose),
    "code" (monospace LTR), "num" (numbers, centred). Defaults to rtl.
    Cell text is escaped unless raw_html=True.
    """
    rows = list(rows)
    n = len(headers)
    dirs = list(col_dirs or ["rtl"] * n)
    if len(dirs) != n:
        raise ValueError("col_dirs must match headers length")
    conv = (lambda s: s) if raw_html else inline
    head = "".join(f'<th class="{d}">{conv(h)}</th>' for h, d in zip(headers, dirs))
    body = ""
    for row in rows:
        if len(row) != n:
            raise ValueError(f"row has {len(row)} cells, expected {n}: {row}")
        body += "<tr>" + "".join(
            f'<td class="{d}">{conv(c)}</td>' for c, d in zip(row, dirs)
        ) + "</tr>"
    cap = f"<caption style='caption-side:top;text-align:right;color:#6B675F;padding:.2rem .4rem'>{esc(caption)}</caption>" if caption else ""
    st.html(
        f'<div class="dlia-table-wrap"><table class="dlia-table">{cap}'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def term_table(rows: Iterable[tuple[str, str, str, str]]) -> None:
    """The canonical 4-column terminology table: المفهوم | English | التفسير | مثال."""
    table(
        ["المفهوم", "English Term", "التفسير", "مثال"],
        [(ar, e, ex, exm) for ar, e, ex, exm in rows],
        ["rtl", "ltr", "rtl", "code"],
    )


def pipeline(steps: Sequence[str], active: int | None = None) -> None:
    """Horizontal LTR process pipeline (spec §24) with an optional highlighted stage."""
    parts = []
    for i, s in enumerate(steps):
        cls = "step active" if i == active else "step"
        parts.append(f'<span class="{cls}">{esc(s)}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="arrow">→</span>')
    st.html(f'<div class="dlia-pipe">{"".join(parts)}</div>')
