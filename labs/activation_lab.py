"""Activation Function Lab — curves, derivatives, gradient decay across depth,
and a live dead-unit counter."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, warning_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table

LAB = Lab(
    id="labs.activation_lab",
    title_ar="معمل دوال التنشيط",
    title_en="Activation Function Lab",
    category="training",
    description_ar="ارسم أي دالة تنشيط ومشتقتها، احسب عامل التدرج عبر عدد من الطبقات عند قيمة z، ومرّر دفعة عشوائية عبر شبكة لترى الإشباع أو الوحدات الميتة طبقةً طبقة.",
    related_lessons=["foundations.activations.derivatives_saturation_dead", "foundations.activations.relu_family"],
)

ACTS = {
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda z: (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z)))),
    "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    "relu": (lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)),
    "leaky_relu": (lambda z: np.where(z > 0, z, 0.1 * z), lambda z: np.where(z > 0, 1.0, 0.1)),
    "elu": (lambda z: np.where(z > 0, z, np.exp(np.minimum(z, 0)) - 1), lambda z: np.where(z > 0, 1.0, np.exp(np.minimum(z, 0)))),
    "gelu": (lambda z: 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3))), None),
}


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    name = st.selectbox("الدالة", list(ACTS), key="al_act")
    f, df = ACTS[name]
    z = np.linspace(-6, 6, 400)
    if df is None:
        h = 1e-4; d = (f(z + h) - f(z - h)) / (2 * h)
    else:
        d = df(z)
    z0 = st.slider("z₀", -6.0, 6.0, 2.0, 0.25, key="al_z0")
    d0 = float(np.interp(z0, z, d))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z, y=f(z), name=f"{name}(z)", line=dict(color="#1F7A78", width=3)))
    fig.add_trace(go.Scatter(x=z, y=d, name=f"{name}'(z)", line=dict(color="#C8473A", dash="dot", width=2)))
    fig.add_trace(go.Scatter(x=[z0], y=[float(f(z0))], mode="markers", marker=dict(size=12, color="#2F6FB5"), name="z₀"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="al_fig")

    st.markdown("### عامل التدرج عبر العمق")
    L = st.slider("عدد الطبقات", 1, 30, 10, key="al_L")
    rows = [(str(k), f"{d0 ** k:.3e}") for k in [1, 2, 5, 10, 20, 30] if k <= L] + [(str(L), f"{d0 ** L:.3e}")]
    table(["الطبقات", f"f'(z₀)^L عند z₀ = {z0}"], rows, ["num", "num"])
    intuition(f"مشتقة {name} عند z₀ = {z0} هي {d0:.4f}. مضروبة في نفسها {L} مرة تعطي {d0 ** L:.2e}: هذا ما يصل إلى الطبقة الأولى إذا كانت كل الوحدات عند نفس النقطة.")
    if d0 < 0.3 and name in ("sigmoid", "tanh"):
        warning_note("إشباع: المشتقة صغيرة. حرّك z₀ نحو الصفر أو اختر ReLU.")
    if d0 == 0.0:
        warning_note("مشتقة صفر: وحدة ميتة (ReLU بمدخل سالب). لا تدرج مهما كان عدد الطبقات.")

    st.markdown("### دفعة عبر شبكة: الإشباع والموت طبقةً طبقة")
    c1, c2, c3 = st.columns(3)
    with c1:
        depth = st.slider("العمق", 1, 12, 6, key="al_depth")
    with c2:
        width = st.slider("العرض", 4, 128, 32, key="al_width")
    with c3:
        scale = st.select_slider("مقياس التهيئة σ_w", options=[0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0], value=0.3, key="al_scale")
    rng = np.random.default_rng(0)
    a = rng.normal(size=(256, 16))
    fan_in = 16
    stats = []
    for i in range(depth):
        W = rng.normal(0, scale, (fan_in, width)); z_ = a @ W; a = f(z_); fan_in = width
        dfun = ACTS[name][1]
        dead = float((np.abs(a) < 1e-6).mean())
        sat = float((np.abs(dfun(z_)) < 0.05).mean()) if dfun else 0.0
        stats.append((f"layer {i + 1}", f"{a.mean():+.3f}", f"{a.std():.3f}", f"{dead:.2f}", f"{sat:.2f}"))
    table(["الطبقة", "متوسط التنشيط", "انحرافه", "نسبة الصفرية (ميت)", "نسبة المشتقة < 0.05 (مشبع)"], stats, ["code", "num", "num", "num", "num"])
    st.caption("غيّر σ_w: صغير جدًا → التنشيطات تتلاشى إلى الصفر عبر العمق؛ كبير جدًا → إشباع (Sigmoid/tanh) أو انفجار (ReLU). التهيئة الجيدة (He: σ = √(2/fan_in)) تحافظ على الانحراف ≈ ثابت.")
    st.code(f"He init for fan_in={width}: sigma = sqrt(2/{width}) = {np.sqrt(2 / width):.3f}   Glorot: sqrt(1/{width}) = {np.sqrt(1 / width):.3f}", language="text")
