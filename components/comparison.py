"""Comparison blocks (spec §26, §40): side-by-side tables and good-vs-bad panels."""

from __future__ import annotations

from typing import Sequence

import streamlit as st

from core.rtl import ColDir, table


def compare_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    col_dirs: Sequence[ColDir] | None = None,
    caption: str | None = None,
) -> None:
    table(headers, rows, col_dirs, caption=caption)


def good_vs_bad(
    good_title: str,
    good_body: str,
    bad_title: str,
    bad_body: str,
    *,
    good_code: str | None = None,
    bad_code: str | None = None,
    verdict_ar: str | None = None,
) -> None:
    """Two panels: correct (green) vs wrong (red) — with icon + label + colour."""
    c_good, c_bad = st.columns(2)
    with c_good:
        with st.container(border=True):
            st.markdown(f"**✅ {good_title}**")
            st.markdown(good_body)
            if good_code:
                st.code(good_code, language="python")
    with c_bad:
        with st.container(border=True):
            st.markdown(f"**❌ {bad_title}**")
            st.markdown(bad_body)
            if bad_code:
                st.code(bad_code, language="python")
    if verdict_ar:
        st.markdown(f"**الحكم:** {verdict_ar}")


def concept_card(title_ar: str, title_en: str, body_ar: str, *, kind: str = "data") -> None:
    """Small definitional card with a semantic colour band."""
    from core.rtl import esc

    st.html(
        f'<div class="dlia-card" style="border-right:5px solid var(--sem-{kind})">'
        f'<h4>{esc(title_ar)} <span class="en" style="font-size:.85em">{esc(title_en)}</span></h4>'
        f'<p>{esc(body_ar)}</p></div>'
    )
