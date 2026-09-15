"""Theme: injects the design-token / RTL stylesheet once per run.

Colours and fonts are configured natively in `.streamlit/config.toml`;
`assets/styles/base.css` adds what native theming cannot express — the
bidirectional typography system, semantic colour classes and reusable
card / callout / table styles.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

STYLE_DIR = Path(__file__).resolve().parent.parent / "assets" / "styles"

# Semantic colour identity (spec §15). Each entry: (Arabic label, English label, icon)
SEMANTIC = {
    "data":  ("بيانات / مدخلات", "Input / Data", "📥"),
    "param": ("معلمة", "Parameter", "🎛️"),
    "hyper": ("معلمة فائقة", "Hyperparameter", "🔧"),
    "loss":  ("خسارة / خطأ", "Loss / Error", "📉"),
    "grad":  ("تدرج", "Gradient", "🧭"),
    "ok":    ("حالة صحيحة", "Correct", "✅"),
    "warn":  ("تحذير", "Warning", "⚠️"),
    "fail":  ("فشل", "Failure", "❌"),
}


@st.cache_data(show_spinner=False)
def _load_css(mtimes: tuple[float, ...]) -> str:
    """Concatenate every stylesheet in assets/styles (sorted). Cached by mtime
    so edits are picked up during development without a server restart."""
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(STYLE_DIR.glob("*.css")))


def inject_css() -> None:
    """Inject the design-token / RTL stylesheet into the page."""
    mtimes = tuple(p.stat().st_mtime for p in sorted(STYLE_DIR.glob("*.css")))
    st.markdown(f"<style>{_load_css(mtimes)}</style>", unsafe_allow_html=True)


def semantic_legend_html() -> str:
    chips = "".join(
        f'<span class="chip sem-{key}">{icon} {en} — {ar}</span>'
        for key, (ar, en, icon) in SEMANTIC.items()
    )
    return f'<div class="dlia-rtl" style="display:flex;flex-wrap:wrap;gap:.4rem">{chips}</div>'
