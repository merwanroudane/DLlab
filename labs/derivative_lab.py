"""Derivative Lab — slope & derivative animation (spec §28 Module 5 labs)."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, practical_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc

LAB = Lab(
    id="labs.derivative_lab",
    title_ar="معمل الميل والمشتقة",
    title_en="Slope & Derivative Lab",
    category="math",
    description_ar="اختر دالة ونقطة، شاهد القاطع يتحول إلى مماس عندما تصغر h، وارسم دالة المشتقة كاملة فوق الأصلية.",
    related_lessons=["foundations.calculus.change_slope", "foundations.calculus.derivative"],
)

FUNCS = {
    "x²": (lambda x: x ** 2, lambda x: 2 * x, "2x"),
    "x³ − 3x": (lambda x: x ** 3 - 3 * x, lambda x: 3 * x ** 2 - 3, "3x² − 3"),
    "sigmoid": (lambda x: 1 / (1 + np.exp(-x)), lambda x: (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x))), "σ(x)(1−σ(x))"),
    "ReLU": (lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float), "1 if x>0 else 0"),
    "e^x": (np.exp, np.exp, "e^x"),
    "ln(x) (x>0)": (lambda x: np.log(np.clip(x, 1e-3, None)), lambda x: 1 / np.clip(x, 1e-3, None), "1/x"),
}


def _svg_secant(f, x0: float, h: float) -> str:
    xs = np.linspace(-3, 3, 120)
    ys = f(xs)
    lo, hi = float(np.nanmin(ys)) - 0.5, float(np.nanmax(ys)) + 0.5
    def sx(x): return 30 + (x + 3) / 6 * 380
    def sy(y): return 170 - (y - lo) / (hi - lo) * 150
    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
    m = (f(x0 + h) - f(x0)) / h
    xa, xb = -3, 3
    ya, yb = f(x0) + m * (xa - x0), f(x0) + m * (xb - x0)
    return (f'<svg viewBox="0 0 440 190" width="100%" style="max-width:440px">'
            f'<path d="{path}" fill="none" stroke="#2F6FB5" stroke-width="2.5"/>'
            f'<line x1="{sx(xa):.1f}" y1="{sy(ya):.1f}" x2="{sx(xb):.1f}" y2="{sy(yb):.1f}" stroke="#C8473A" stroke-width="2" stroke-dasharray="5 3"/>'
            f'<circle cx="{sx(x0):.1f}" cy="{sy(f(x0)):.1f}" r="5" fill="#1F7A78"/>'
            f'<circle cx="{sx(x0 + h):.1f}" cy="{sy(f(x0 + h)):.1f}" r="5" fill="#C8473A"/>'
            f'<text x="220" y="185" text-anchor="middle" font-size="12" font-family="JetBrains Mono, monospace">h = {h}   secant slope = {m:.4f}</text></svg>')


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    name = st.selectbox("الدالة", list(FUNCS), key="dl_func")
    f, df, dname = FUNCS[name]
    x0 = st.slider("النقطة x₀", -2.5, 2.5, 1.0, 0.1, key="dl_x0")
    intuition("اضغط ▶: في كل إطار تصغر h إلى النصف؛ راقب ميل القاطع يستقر عند قيمة المشتقة.")
    hs = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.01, 0.001]
    frames = []
    for h in hs:
        m = (f(x0 + h) - f(x0)) / h
        frames.append(Frame(_svg_secant(f, x0, h), caption(f"`h = {h}` ← ميل القاطع `{m:.4f}`؛ المشتقة الحقيقية `{float(df(x0)):.4f}`؛ الفرق `{abs(m - float(df(x0))):.4f}`."),
                            action=f"h = {h}", values=[("secant slope", "", f"{m:.4f}"), ("f'(x0)", "", f"{float(df(x0)):.4f}")]))
    animation_player("deriv_anim", frames, title_ar=f"القاطع → المماس عند x₀ = {x0}", interval_ms=900)

    st.markdown("### دالة المشتقة كاملة")
    xs = np.linspace(-3, 3, 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=f(xs), name=f"f(x) = {name}", line=dict(color="#2F6FB5", width=3)))
    fig.add_trace(go.Scatter(x=xs, y=df(xs), name=f"f'(x) = {dname}", line=dict(color="#C8473A", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=[x0], y=[float(f(x0))], mode="markers", marker=dict(size=11, color="#1F7A78"), name="x₀"))
    fig.add_hline(y=0, line=dict(color="#B9B2A6", width=1))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="dl_fig")
    st.markdown("اقرأ المنحنى الأحمر: حيث يكون موجبًا تصعد الزرقاء، حيث سالب تهبط، حيث صفر قمة/قاع.")
    if name == "sigmoid":
        practical_note("لاحظ أن f'(x) لا تتجاوز 0.25 وتقترب من الصفر بعيدًا عن المركز: هذا هو **الإشباع** وسبب تلاشي التدرج مع Sigmoid في الشبكات العميقة.")
    if name == "ReLU":
        practical_note("المشتقة 0 لكل المدخلات السالبة: وحدة ReLU بمدخل سالب دائمًا لا تتعلم أبدًا (**Dead ReLU**).")
