"""Scaling Lab — see how scaling reshapes the loss surface and the descent path."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, warning_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.datasets import loan_default

LAB = Lab(
    id="labs.scaling_lab",
    title_ar="معمل التحجيم",
    title_en="Scaling Lab",
    category="data",
    description_ar="قارن خصائص خامًا وموحدة ومطبّعة، وشاهد كيف يغيّر التحجيم شكل سطح الخسارة ومسار الانحدار التدريجي ومعدل التعلم الذي ينجح.",
    related_lessons=["foundations.prep.scaling", "foundations.calculus.partial_gradient"],
)


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    df = loan_default().dropna()
    f1 = st.selectbox("الخاصية 1", ["income", "age", "num_late_payments", "debt_ratio"], index=0, key="sl_f1")
    f2 = st.selectbox("الخاصية 2", ["income", "age", "num_late_payments", "debt_ratio"], index=1, key="sl_f2")
    method = st.segmented_control("التحجيم", ["خام", "توحيد قياسي (z-score)", "تطبيع min-max"], default="خام", key="sl_method")
    X = df[[f1, f2]].to_numpy(float)
    if method == "توحيد قياسي (z-score)":
        Xs = (X - X.mean(0)) / X.std(0)
    elif method == "تطبيع min-max":
        Xs = (X - X.min(0)) / (X.max(0) - X.min(0))
    else:
        Xs = X.copy()
    rows = [(f, f"{Xs[:, i].min():.3g}", f"{Xs[:, i].max():.3g}", f"{Xs[:, i].mean():.3g}", f"{Xs[:, i].std():.3g}") for i, f in enumerate([f1, f2])]
    table(["الخاصية", "min", "max", "mean", "std"], rows, ["code", "num", "num", "num", "num"])
    intuition("انظر إلى عمود std: بالخام قد تختلف الخاصيتان بآلاف المرات؛ بعد التوحيد كلاهما 1.")

    st.markdown("### سطح خسارة انحدار خطي على الخاصيتين ومسار النزول")
    y = df["defaulted"].to_numpy(float)
    y = (y - y.mean()) / (y.std() + 1e-9)
    Xc = Xs - Xs.mean(0)
    eta = st.select_slider("معدل التعلم η", options=[1e-9, 1e-7, 1e-5, 1e-3, 0.01, 0.1, 0.5, 1.0], value=0.1, key="sl_eta")
    steps = st.slider("الخطوات", 5, 100, 30, key="sl_steps")
    w = np.zeros(2); path = [w.copy()]; diverged = False
    for _ in range(steps):
        g = 2 * Xc.T @ (Xc @ w - y) / len(y)
        w = w - eta * g
        if not np.all(np.isfinite(w)) or np.abs(w).max() > 1e6:
            diverged = True; break
        path.append(w.copy())
    path = np.array(path)
    w_opt = np.linalg.lstsq(Xc, y, rcond=None)[0]
    span = max(np.abs(w_opt).max() * 2.5, np.abs(path).max() * 1.2, 1e-6)
    ws = np.linspace(-span, span, 60)
    W0, W1 = np.meshgrid(ws, ws)
    Z = np.array([[np.mean((Xc @ np.array([a, b]) - y) ** 2) for a in ws] for b in ws])
    fig = go.Figure(go.Contour(x=ws, y=ws, z=np.log1p(Z - Z.min()), colorscale="YlGnBu", ncontours=25, showscale=False))
    fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines+markers", line=dict(color="#1F7A78", width=2), name="مسار GD"))
    fig.add_trace(go.Scatter(x=[w_opt[0]], y=[w_opt[1]], mode="markers", marker=dict(size=13, color="#C8473A", symbol="star"), name="الحل الأمثل"))
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=20, b=10), xaxis_title=f"w[{f1}]", yaxis_title=f"w[{f2}]", legend=dict(orientation="h"), paper_bgcolor="#FFFDF9")
    st.plotly_chart(fig, width="stretch", key="sl_fig")
    if diverged:
        st.error("🔥 تباعد. مع خصائص خام بمقاييس كبيرة يحتاج معدل التعلم إلى أن يكون صغيرًا جدًا (1e-7 أو أقل) — ويصبح عندها الاتجاه الآخر بطيئًا جدًا.", icon="❌")
    else:
        st.code(f"final w = {path[-1].round(4)}   optimal = {w_opt.round(4)}   distance = {np.linalg.norm(path[-1] - w_opt):.4f}", language="text")
    if method == "خام":
        warning_note("بالخام سطح الخسارة وادٍ ضيق ممدود: لا يوجد معدل تعلم واحد يناسب الاتجاهين. جرّب التوحيد ثم أعد نفس η.")
