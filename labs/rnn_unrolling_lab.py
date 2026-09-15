"""RNN Unrolling Lab — an Elman cell steps through a short sequence; each
frame shows x_t, h_{t-1}, the weighted sums and h_t, plus the BPTT
gradient-magnitude view for vanishing / exploding (spec §41)."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, practical_note
from components.diagram import svg_arrow, svg_box, svg_defs, svg_text
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.rnn import bptt_gradient_norms, random_weights, rnn_forward

LAB = Lab(
    id="labs.rnn_unrolling_lab",
    title_ar="معمل نشر RNN عبر الزمن",
    title_en="RNN Unrolling Lab",
    category="sequence",
    description_ar="تسلسل قصير يدخل خلية RNN خطوةً زمنيةً خطوة: المدخل x_t، الحالة السابقة h_{t−1}، المجموع الموزون، والحالة الجديدة h_t — ثم كيف يتلاشى أو ينفجر التدرج عبر الزمن (BPTT).",
    related_lessons=["course.w10.sequences_hidden_state", "course.w10.rnn_bptt_vanishing"],
)


def _unrolled_svg(steps: list[dict], active: int) -> str:
    T = len(steps); cell = 110
    s = [f'<svg viewBox="0 0 {40 + T * cell} 200" width="100%" style="max-width:{40 + T * cell}px">', svg_defs()]
    for t, stp in enumerate(steps):
        x = 20 + t * cell; on = t == active; done = t < active
        fill = "#E3F3F0" if on else ("#F1EFEA" if done else "#FFFDF9")
        s.append(svg_box(x, 70, 80, 50, f"RNN t={t}", fill, stroke="#1F7A78" if on else "#B9B2A6", font=12, bold=on))
        s.append(svg_text(x + 40, 40, f"x{t} = {stp['x'][0]:+.1f}", size=11, mono=True, color="#2F6FB5"))
        s.append(svg_arrow(x + 40, 48, x + 40, 68, color="#2F6FB5"))
        hval = f"h{t} = [{', '.join(f'{v:+.2f}' for v in stp['h'][:2])}{'…' if len(stp['h']) > 2 else ''}]" if (on or done) else f"h{t} = ?"
        s.append(svg_text(x + 40, 165, hval, size=10, mono=True, color="#C8473A" if on else "#6B675F"))
        s.append(svg_arrow(x + 40, 122, x + 40, 150, color="#C8473A"))
        if t < T - 1:
            s.append(svg_arrow(x + 82, 95, x + cell - 2, 95, color="#C8473A" if (on or done) else "#B9B2A6"))
            s.append(svg_text(x + cell - 14, 88, "Wh", size=9, mono=True, color="#6B675F"))
    s.append(svg_text(20 + T * cell / 2, 190, "same Wx, Wh, b at every timestep — the network is one cell, drawn T times", size=11, color="#6B675F"))
    s.append("</svg>")
    return "".join(s)


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    c1, c2, c3 = st.columns(3)
    with c1:
        seq_txt = st.text_input("التسلسل x (أرقام بفواصل)", "0.5, -1.0, 2.0, 0.0, 1.5", key="ru_seq")
    with c2:
        H = st.select_slider("حجم الحالة المخفية H", options=[2, 3, 4, 8], value=3, key="ru_h")
    with c3:
        scale = st.select_slider("مقياس الأوزان", options=[0.3, 0.6, 1.0, 1.5], value=0.6, key="ru_scale")
    try:
        xs = np.array([float(v) for v in seq_txt.replace("،", ",").split(",") if v.strip()])
    except ValueError:
        xs = np.array([0.5, -1.0, 2.0, 0.0, 1.5])
    W = random_weights("rnn", 1, int(H), seed=0, scale=float(scale))
    steps = rnn_forward(xs, **W)
    st.code(f"h_t = tanh(x_t·Wx + h_(t-1)·Wh + b)     Wx: (1, {H})  Wh: ({H}, {H})  b: ({H},)  -> params = {1 * H + H * H + H}   (independent of T = {len(xs)})", language="text")
    frames = []
    for stp in steps:
        t = stp["t"]
        frames.append(Frame(_unrolled_svg(steps, t), caption(f"**الخطوة الزمنية t = {t}**: المدخل `x{t} = {stp['x'][0]:+.2f}`، الحالة السابقة `h{t - 1} = {np.round(stp['h_prev'], 2).tolist()}`. المجموع `z = x·Wx + h·Wh + b = {np.round(stp['z'], 2).tolist()}` ثم `h{t} = tanh(z) = {np.round(stp['h'], 2).tolist()}`. **نفس الأوزان** في كل خطوة؛ ما يتغير هو x وh."),
                            action=f"t = {t}", equation=f"h{t} = tanh(x{t}·Wx + h{t - 1}·Wh + b)", values=[("h[0]", f"{stp['h_prev'][0]:+.2f}", f"{stp['h'][0]:+.2f}")]))
    animation_player("ru_anim", frames, title_ar="خلية واحدة، تُرسم T مرات", interval_ms=1500)
    table(["t", "x_t", "h_(t−1)", "h_t"], [(str(s_["t"]), f"{s_['x'][0]:+.2f}", str(np.round(s_["h_prev"], 2).tolist()), str(np.round(s_["h"], 2).tolist())) for s_ in steps], ["num", "num", "code", "code"])
    intuition("h_t «يلخّص» كل ما مضى: x_0…x_t عبر سلسلة من tanh. بمقياس أوزان كبير (1.5) تتشبع tanh وتنسى الحالة تفاصيلها؛ بمقياس صغير تتلاشى إسهامات المدخلات القديمة. كلاهما وجه للمشكلة التي يعالجها LSTM/GRU.")
    st.markdown("### BPTT: كم يصل من التدرج إلى الخطوات القديمة؟")
    c4, c5 = st.columns(2)
    with c4:
        T = st.slider("طول التسلسل T", 5, 40, 20, key="ru_T")
    with c5:
        rho = st.select_slider("نصف قطر طيفي لـ Wh", options=[0.5, 0.9, 1.0, 1.2, 3.0], value=0.5, key="ru_rho")
    fig = go.Figure()
    for r_, color in ((0.5, "#2F6FB5"), (1.0, "#1F7A78"), (3.0, "#C8473A")):
        norms = bptt_gradient_norms(int(T), wh_scale=r_)
        fig.add_trace(go.Scatter(x=list(range(T, 0, -1)), y=norms, name=f"ρ(Wh) = {r_}", line=dict(color=color, width=2.5 if r_ == rho else 1.5, dash=None if r_ == rho else "dot")))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="timestep t (gradient from t = T back to t)", yaxis_title="‖∂h_T / ∂h_t‖", yaxis_type="log", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="ru_bptt")
    practical_note("التدرج من الخطوة T إلى الخطوة t حاصل ضرب T−t يعقوبيًا (كل واحد diag(1−h²)·Whᵀ). بنصف قطر < 1 يتلاشى أسيًا (المحور لوغاريتمي)، وبنصف قطر > 1 قد ينفجر حتى تكبحه tanh. العلاج: LSTM/GRU (مسار جمع بدل ضرب)، قصّ التدرج، تسلسلات أقصر.")
