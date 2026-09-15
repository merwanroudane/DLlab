"""Equation breakdown (spec §11): every equation is followed by a symbol
table, a meaning line and a numerical example — never a bare formula."""

from __future__ import annotations

from typing import Sequence

import streamlit as st

from core.rtl import esc


def equation(
    latex: str,
    symbols: Sequence[tuple[str, str]] | None = None,
    *,
    meaning_ar: str | None = None,
    example_ar: str | None = None,
    dl_link_ar: str | None = None,
    title_ar: str | None = None,
) -> None:
    """Render an equation with its breakdown.

    symbols: [(latex_symbol, arabic_explanation), ...]
    """
    with st.container(border=True):
        if title_ar:
            st.markdown(f"**∑ {title_ar}**")
        st.latex(latex)
        if symbols:
            rows = "".join(
                f'<div class="row"><div class="sym">${esc(sym)}$</div><div>{esc(txt)}</div></div>'
                for sym, txt in symbols
            )
            # KaTeX in st.markdown handles inline $...$; we render symbol names via markdown
            # to get real math glyphs, so build a markdown list instead of raw HTML here.
            lines = []
            for sym, txt in symbols:
                lines.append(f"- ${sym}$ — {txt}")
            st.markdown("**تفكيك الرموز:**")
            st.markdown("\n".join(lines))
        if meaning_ar:
            st.markdown(f"**المعنى:** {meaning_ar}")
        if example_ar:
            st.markdown(f"**مثال رقمي:** {example_ar}")
        if dl_link_ar:
            st.markdown(f"**الصلة بالتعلم العميق:** {dl_link_ar}")


def worked_steps(steps: Sequence[tuple[str, str]], title_ar: str = "الحساب خطوة بخطوة") -> None:
    """Numbered step-by-step calculation: [(arabic_description, latex_or_code), ...]."""
    st.markdown(f"**🧮 {title_ar}**")
    for i, (desc, expr) in enumerate(steps, start=1):
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown(f"**{i}.** {desc}")
        with c2:
            if expr.startswith("`"):
                st.code(expr.strip("`"), language="python")
            else:
                st.latex(expr)
