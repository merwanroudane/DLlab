"""Diagram wrapper (spec §25): every diagram ships with a title, "what am I
looking at", the drawing, an optional legend, "how to read it" and a key
takeaway. SVG is authored inline (LTR) and never clips Arabic labels
because labels live in the surrounding HTML, not inside the SVG."""

from __future__ import annotations

from typing import Sequence

import streamlit as st

from core.rtl import esc


def diagram(
    title_ar: str,
    svg: str,
    *,
    what_ar: str,
    how_ar: str,
    takeaway_ar: str,
    legend: Sequence[tuple[str, str]] | None = None,   # [(colour, label_ar), ...]
    title_en: str | None = None,
    max_width: int = 720,
) -> None:
    legend_html = ""
    if legend:
        items = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:.35rem;margin-inline-end:.9rem">'
            f'<span style="width:.9rem;height:.9rem;border-radius:3px;background:{esc(color)};display:inline-block"></span>'
            f"{esc(label)}</span>"
            for color, label in legend
        )
        legend_html = f'<div class="dlia-rtl" style="margin:.4rem 0;font-size:.92rem">{items}</div>'
    en = f' <span class="en" style="font-size:.85em">{esc(title_en)}</span>' if title_en else ""
    st.html(
        f'<div class="dlia-card">'
        f'<h4>🗺️ {esc(title_ar)}{en}</h4>'
        f'<p class="muted"><b>ماذا أشاهد؟</b> {esc(what_ar)}</p>'
        f'<div style="direction:ltr;text-align:center;max-width:{max_width}px;margin:.4rem auto;overflow-x:auto">{svg}</div>'
        f"{legend_html}"
        f'<p><b>كيف أقرؤه؟</b> {esc(how_ar)}</p>'
        f'<p><b>🎯 الخلاصة:</b> {esc(takeaway_ar)}</p>'
        f"</div>"
    )


# ---------------------------------------------------------------- helpers
def svg_box(x: float, y: float, w: float, h: float, text: str, fill: str, *,
            stroke: str = "#B9B2A6", font: int = 14, rx: int = 10, text_color: str = "#2B2A28",
            bold: bool = False) -> str:
    weight = "600" if bold else "500"
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'
        f'<text x="{x + w / 2}" y="{y + h / 2 + font * 0.35}" text-anchor="middle" '
        f'font-family="Inter, Segoe UI, sans-serif" font-size="{font}" font-weight="{weight}" fill="{text_color}">{esc(text)}</text>'
    )


def svg_arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#6B675F", width: float = 2) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" '
        f'marker-end="url(#arrowhead)"/>'
    )


def svg_defs() -> str:
    return (
        '<defs><marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
        '<path d="M0,0 L8,4 L0,8 z" fill="#6B675F"/></marker></defs>'
    )


def svg_text(x: float, y: float, text: str, *, size: int = 13, color: str = "#2B2A28",
             anchor: str = "middle", bold: bool = False, mono: bool = False) -> str:
    fam = "JetBrains Mono, Consolas, monospace" if mono else "Inter, Segoe UI, sans-serif"
    w = "600" if bold else "400"
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{fam}" font-size="{size}" '
            f'font-weight="{w}" fill="{color}">{esc(text)}</text>')
