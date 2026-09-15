"""Framework Visual Gallery frames (spec §22 of the module list).

Keras / TensorFlow / PyTorch have no GUI of their own, so the gallery never
invents one. Each frame is an *educational recreation* of a real surface —
a notebook cell, a console, a TensorBoard panel — built from **real output
produced by this project's pinned versions**, and it is labelled as such.

Every frame answers the six questions of the spec: where are we? what do we
see? where is the code? where is the output? what do the numbers mean? what
should the researcher notice?
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field

import streamlit as st

from core.rtl import esc

LABEL_AR = "إعادة بناء تعليمية"
LABEL_EN = "Educational recreation"


@dataclass
class Frame6:
    where_ar: str          # أين نحن؟
    what_ar: str           # ماذا نرى؟
    code_where_ar: str     # أين الكود؟
    output_where_ar: str   # أين المخرج؟
    numbers_ar: str        # ماذا تعني الأرقام؟
    notice_ar: str         # ما الذي يجب أن يلاحظه الباحث؟
    extra_ar: list[str] = field(default_factory=list)


def _badge(kind: str) -> str:
    return (f'<div style="display:flex;justify-content:space-between;align-items:center;background:#2B2A28;color:#F1EFEA;'
            f'padding:.3rem .7rem;border-radius:10px 10px 0 0;font-size:.8rem">'
            f'<span class="en">{esc(kind)}</span>'
            f'<span style="background:#D9A21B;color:#2B2A28;border-radius:6px;padding:.1rem .5rem;font-weight:600">{LABEL_EN} · {LABEL_AR}</span></div>')


def notebook_cell(code: str, output: str, *, n: int = 1, kind: str = "Jupyter / Colab notebook", stderr: str | None = None, image_html: str | None = None) -> str:
    """HTML for a recreated notebook cell (code + text output [+ figure])."""
    code_html = _html.escape(code)
    out_html = _html.escape(output)
    parts = [
        _badge(kind),
        '<div dir="ltr" style="text-align:left;border:1px solid #B9B2A6;border-top:0;border-radius:0 0 10px 10px;background:#FFFFFF;font-family:JetBrains Mono,Consolas,monospace;font-size:12.5px">',
        f'<div style="display:flex"><div style="width:62px;color:#303F9F;padding:8px 6px;text-align:right;flex:none">In&nbsp;[{n}]:</div>'
        f'<pre style="margin:0;padding:8px 10px;flex:1;background:#F7F7F7;border-left:3px solid #C8D3F5;white-space:pre-wrap;line-height:1.45">{code_html}</pre></div>',
    ]
    if output:
        parts.append(f'<div style="display:flex"><div style="width:62px;color:#D84315;padding:8px 6px;text-align:right;flex:none">Out:</div>'
                     f'<pre style="margin:0;padding:8px 10px;flex:1;white-space:pre-wrap;line-height:1.45;color:#2B2A28">{out_html}</pre></div>')
    if stderr:
        parts.append(f'<div style="display:flex"><div style="width:62px;flex:none"></div>'
                     f'<pre style="margin:0 0 6px 0;padding:8px 10px;flex:1;white-space:pre-wrap;line-height:1.45;color:#B71C1C;background:#FDECEA">{_html.escape(stderr)}</pre></div>')
    if image_html:
        parts.append(f'<div style="display:flex"><div style="width:62px;flex:none"></div><div style="padding:6px 10px;flex:1">{image_html}</div></div>')
    parts.append("</div>")
    return "".join(parts)


def console(text: str, *, kind: str = "Terminal / console", prompt: str = "$ python train.py") -> str:
    return (_badge(kind) + '<div dir="ltr" style="text-align:left;background:#1E1E1E;color:#D4D4D4;border-radius:0 0 10px 10px;padding:10px 12px;'
            f'font-family:JetBrains Mono,Consolas,monospace;font-size:12.5px;line-height:1.5;white-space:pre-wrap"><span style="color:#6A9955">{_html.escape(prompt)}</span>\n{_html.escape(text)}</div>')


def gallery_item(title_ar: str, title_en: str, frame_html: str, six: Frame6, *, key: str | None = None) -> None:
    """Render one gallery entry: the recreated frame + the six-question reading guide."""
    st.markdown(f"### 🖼️ {title_ar} <span class='en' style='font-size:.75em;color:#6B675F'>{esc(title_en)}</span>", unsafe_allow_html=True)
    if frame_html:
        st.html(frame_html)
    rows = [("📍 أين نحن؟", six.where_ar), ("👀 ماذا نرى؟", six.what_ar), ("⌨️ أين الكود؟", six.code_where_ar), ("📤 أين المخرج؟", six.output_where_ar), ("🔢 ماذا تعني الأرقام؟", six.numbers_ar), ("🎯 ما الذي يجب أن يلاحظه الباحث؟", six.notice_ar)]
    with st.expander("كيف أقرأ هذه اللقطة؟ (الأسئلة الستة)", expanded=True, icon=":material/visibility:"):
        for q, a in rows:
            st.markdown(f"**{q}** {a}")
        for e in six.extra_ar:
            st.markdown(f"- {e}")


def svg_line_chart(series: list[tuple[str, list[float], str]], *, title: str = "", width: int = 520, height: int = 250,
                   xlabel: str = "epoch", ylabel: str = "loss", start_at: int = 1, legend: bool = True) -> str:
    """Small self-contained SVG line chart (matplotlib-like) for recreated
    notebook figures and TensorBoard panels. series: [(name, values, colour)]."""
    import math

    ml, mr, mt, mb = 48, 14, 28 if title else 12, 34
    pw, ph = width - ml - mr, height - mt - mb
    ys = [v for _, vals, _ in series for v in vals if not (v is None or math.isnan(v))]
    n = max(len(vals) for _, vals, _ in series)
    lo, hi = (min(ys), max(ys)) if ys else (0.0, 1.0)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    pad = (hi - lo) * 0.08; lo -= pad; hi += pad

    def X(i):
        return ml + (pw * i / max(n - 1, 1))

    def Y(v):
        return mt + ph - (v - lo) / (hi - lo) * ph

    s = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;font-family:Inter,Segoe UI,sans-serif;background:#FFF">']
    if title:
        s.append(f'<text x="{width / 2}" y="16" text-anchor="middle" font-size="12" font-weight="600" fill="#2B2A28">{esc(title)}</text>')
    s.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#2B2A28" stroke-width="1"/>')
    for k in range(5):
        v = lo + (hi - lo) * k / 4; y = Y(v)
        s.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" stroke="#E5E0D6" stroke-width="1"/>')
        s.append(f'<text x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="#2B2A28">{v:.2f}</text>')
    step = max(1, n // 6)
    for i in range(0, n, step):
        s.append(f'<text x="{X(i):.1f}" y="{mt + ph + 14}" text-anchor="middle" font-size="10" fill="#2B2A28">{i + start_at}</text>')
    s.append(f'<text x="{ml + pw / 2}" y="{height - 4}" text-anchor="middle" font-size="11" fill="#2B2A28">{esc(xlabel)}</text>')
    s.append(f'<text x="12" y="{mt + ph / 2}" text-anchor="middle" font-size="11" fill="#2B2A28" transform="rotate(-90 12 {mt + ph / 2})">{esc(ylabel)}</text>')
    for name, vals, color in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
    if legend:
        for j, (name, _, color) in enumerate(series):
            yy = mt + 12 + j * 14
            s.append(f'<line x1="{ml + pw - 96}" y1="{yy}" x2="{ml + pw - 78}" y2="{yy}" stroke="{color}" stroke-width="2"/>')
            s.append(f'<text x="{ml + pw - 72}" y="{yy + 4}" font-size="10" fill="#2B2A28">{esc(name)}</text>')
    s.append("</svg>")
    return "".join(s)
