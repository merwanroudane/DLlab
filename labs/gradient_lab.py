"""Gradient Lab — gradient direction & descent path on a 2-D loss surface."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, warning_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table

LAB = Lab(
    id="labs.gradient_lab",
    title_ar="معمل التدرج",
    title_en="Gradient Lab",
    category="math",
    description_ar="سطح خسارة بمعلمتين: اختر نقطة البداية ومعدل التعلم، وشاهد اتجاه التدرج ومسار النزول خطوة بخطوة — وكيف يتذبذب أو يتباعد.",
    related_lessons=["foundations.calculus.partial_gradient"],
)

SURFACES = {
    "وعاء دائري": (lambda w, b: (w - 2) ** 2 + (b + 1) ** 2, lambda w, b: np.array([2 * (w - 2), 2 * (b + 1)])),
    "وعاء إهليلجي (مقاييس مختلفة)": (lambda w, b: (w - 2) ** 2 + 8 * (b + 1) ** 2, lambda w, b: np.array([2 * (w - 2), 16 * (b + 1)])),
    "نقطة سرج": (lambda w, b: w ** 2 - b ** 2, lambda w, b: np.array([2 * w, -2 * b])),
    "وعاء مع حد محلي (روزنبروك مبسط)": (lambda w, b: (1 - w) ** 2 + 5 * (b - w ** 2) ** 2,
                                        lambda w, b: np.array([-2 * (1 - w) - 20 * w * (b - w ** 2), 10 * (b - w ** 2)])),
}


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    name = st.selectbox("سطح الخسارة L(w, b)", list(SURFACES), key="gl_surface")
    L, grad = SURFACES[name]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        w0 = st.slider("w₀", -3.0, 5.0, 4.0, 0.25, key="gl_w0")
    with c2:
        b0 = st.slider("b₀", -4.0, 4.0, 3.0, 0.25, key="gl_b0")
    with c3:
        eta = st.select_slider("معدل التعلم η", options=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0], value=0.1, key="gl_eta")
    with c4:
        steps = st.slider("عدد الخطوات", 1, 60, 20, key="gl_steps")
    intuition("السهم الأحمر = التدرج (أسرع صعود). المسار الأخضر = خطوات −η∇L. في الوعاء الإهليلجي لاحظ التعرج؛ في السرج لاحظ الهروب؛ وبمعدل كبير لاحظ التباعد.")

    w, b = w0, b0
    path = [(w, b, L(w, b))]
    diverged = False
    for _ in range(steps):
        g = grad(w, b)
        w, b = w - eta * g[0], b - eta * g[1]
        if not np.isfinite(w) or abs(w) > 1e6 or abs(b) > 1e6:
            diverged = True
            break
        path.append((w, b, L(w, b)))
    ws = np.linspace(-3, 5, 80); bs = np.linspace(-4, 4, 80)
    W, B = np.meshgrid(ws, bs)
    Z = L(W, B)
    Z = np.clip(Z, None, np.percentile(Z, 97))
    fig = go.Figure(go.Contour(x=ws, y=bs, z=Z, colorscale="YlGnBu", ncontours=25, showscale=False))
    px, py = [p[0] for p in path], [p[1] for p in path]
    fig.add_trace(go.Scatter(x=px, y=py, mode="lines+markers", line=dict(color="#1F7A78", width=2), marker=dict(size=6), name="مسار النزول"))
    g0 = grad(w0, b0); s = 0.1 / max(1e-9, np.linalg.norm(g0)) * 2
    fig.add_trace(go.Scatter(x=[w0, w0 + s * g0[0]], y=[b0, b0 + s * g0[1]], mode="lines", line=dict(color="#C8473A", width=3), name="∇L عند البداية"))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10), xaxis=dict(range=[-3, 5], title="w"), yaxis=dict(range=[-4, 4], title="b"), legend=dict(orientation="h"), paper_bgcolor="#FFFDF9")
    st.plotly_chart(fig, width="stretch", key="gl_fig")
    if diverged:
        st.error("🔥 تباعد: القيم انفجرت. معدل التعلم أكبر من أن يحتمله انحناء هذا السطح. قلّله.", icon="❌")
    rows = [(str(i), f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.5f}", f"{np.linalg.norm(grad(p[0], p[1])):.4f}") for i, p in enumerate(path[: min(len(path), 12)])]
    table(["step", "w", "b", "L", "‖∇L‖"], rows, ["num", "num", "num", "num", "num"], caption="أول 12 خطوة")
    if name.startswith("نقطة سرج"):
        warning_note("عند السرج التدرج صفر لكنه ليس حدًا أدنى: أي إزاحة في اتجاه b تهرب إلى −∞. ابدأ من b₀ = 0 بالضبط لترى «الالتصاق» ثم غيّره قليلًا.")
    if name.startswith("وعاء إهليلجي"):
        warning_note("المقاييس المختلفة (8 مقابل 1) تجعل معدل تعلم مناسبًا لاتجاه كبيرًا جدًا للآخر. هذا سبب تحجيم الخصائص، وسبب المحسّنات التكيفية مثل Adam.")
