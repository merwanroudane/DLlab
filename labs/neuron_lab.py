"""Neuron Lab — set weights/bias/activation, see z, a and the decision boundary."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, practical_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table

LAB = Lab(
    id="labs.neuron_lab",
    title_ar="معمل الخلية العصبية",
    title_en="Neuron Lab",
    category="training",
    description_ar="خلية بمدخلين: حرّك الوزنين والانحياز واختر التنشيط، وشاهد z وa لملاحظة واحدة، وحد القرار على المستوى، ومساهمة كل مدخل.",
    related_lessons=["foundations.neuron.weighted_sum_bias", "foundations.neuron.neuron_perceptron", "foundations.neuron.activation_intro"],
)

ACTS = {
    "linear": (lambda z: z, "a = z"),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), "a = 1/(1+e^{-z})"),
    "tanh": (np.tanh, "a = tanh(z)"),
    "relu": (lambda z: np.maximum(0, z), "a = max(0, z)"),
}


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        w1 = st.slider("w₁", -3.0, 3.0, 1.5, 0.1, key="nl_w1")
    with c2:
        w2 = st.slider("w₂", -3.0, 3.0, -1.0, 0.1, key="nl_w2")
    with c3:
        b = st.slider("b", -3.0, 3.0, 0.5, 0.1, key="nl_b")
    with c4:
        act = st.selectbox("التنشيط", list(ACTS), index=1, key="nl_act")
    f, formula = ACTS[act]
    c1, c2 = st.columns(2)
    with c1:
        x1 = st.slider("المدخل x₁", -3.0, 3.0, 1.0, 0.1, key="nl_x1")
    with c2:
        x2 = st.slider("المدخل x₂", -3.0, 3.0, 0.5, 0.1, key="nl_x2")
    z = w1 * x1 + w2 * x2 + b; a = float(f(z))
    table(["الخطوة", "الحساب", "القيمة"],
          [("مساهمة x₁", f"w₁·x₁ = {w1:.1f}×{x1:.1f}", f"{w1 * x1:+.3f}"), ("مساهمة x₂", f"w₂·x₂ = {w2:.1f}×{x2:.1f}", f"{w2 * x2:+.3f}"), ("الانحياز", "b", f"{b:+.3f}"),
           ("المجموع الموزون", "z = Σ + b", f"{z:+.3f}"), ("التنشيط", formula, f"{a:+.3f}")], ["rtl", "code", "num"])
    intuition("غيّر التنشيط وراقب السطر الأخير فقط: z ثابت، a يتغير. ثم حرّك b: z ينزاح، والحد الفاصل (الخط) ينزلق موازيًا لنفسه.")
    g = np.linspace(-3, 3, 90); GX, GY = np.meshgrid(g, g)
    Z = w1 * GX + w2 * GY + b; A = f(Z)
    fig = go.Figure(go.Contour(x=g, y=g, z=A, colorscale="RdBu_r" if act != "relu" else "YlGnBu", contours=dict(showlabels=True), showscale=True))
    if abs(w2) > 1e-6:
        fig.add_trace(go.Scatter(x=g, y=(-b - w1 * g) / w2, mode="lines", line=dict(color="#1F7A78", width=3), name="z = 0"))
    fig.add_trace(go.Scatter(x=[x1], y=[x2], mode="markers", marker=dict(size=14, color="#C8473A", symbol="x"), name="(x₁, x₂)"))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="x₁", yaxis_title="x₂", yaxis=dict(scaleanchor="x"), paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="nl_fig")
    st.caption("الألوان = قيمة a على كامل المستوى. الخط الأخضر حيث z = 0 (Sigmoid = 0.5، ReLU يبدأ). الأوزان تحدد اتجاه الخط، والانحياز موضعه.")
    practical_note("الخلية الواحدة ترسم دائمًا **خطًا** واحدًا مهما كان التنشيط؛ التنشيط يغيّر كيف تُترجم المسافة عن الخط إلى مخرج. الحدود المنحنية تحتاج طبقة مخفية (معمل بناء الشبكة).")
