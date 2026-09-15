"""Regularization Lab — L2 strength sweep: weights, curves, boundary smoothness."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, warning_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.tinynet import TinyNet, make_moons, train

LAB = Lab(
    id="labs.regularization_lab",
    title_ar="معمل التنظيم (L2)",
    title_en="Regularization Lab (L2)",
    category="training",
    description_ar="مسح لقوة L2: لكل λ شاهد خسارة التدريب والتحقق، معيار الأوزان، ونعومة حد القرار على شبكة كبيرة ببيانات قليلة.",
    related_lessons=["foundations.regularization.l1_l2_weight_decay", "foundations.regularization.augmentation_simplification"],
)


@st.cache_data(max_entries=32, show_spinner=False)
def _run(l2: float, n: int, noise: float, seed: int):
    X, y = make_moons(n + 300, noise=noise, seed=seed)
    net = TinyNet([2, 64, 64, 1], "binary", seed=seed)
    h = train(net, X[:n], y[:n], X_val=X[n:], y_val=y[n:], epochs=250, batch_size=16, lr=0.02, optimizer="adam", l2=l2, seed=seed)
    wnorm = float(np.sqrt(sum((W ** 2).sum() for W in net.W)))
    g = np.linspace(-2, 3, 70); GX, GY = np.meshgrid(g, g)
    P = net.predict(np.column_stack([GX.ravel(), GY.ravel()]).astype(np.float32)).reshape(GX.shape)
    return h["loss"][-1], h["val_loss"][-1], h["val_metric"][-1], wnorm, g, P, X[:n], y[:n]


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    c1, c2 = st.columns(2)
    with c1:
        n = st.select_slider("حجم التدريب n", options=[40, 80, 160], value=80, key="rl_n")
    with c2:
        noise = st.select_slider("ضوضاء", options=[0.2, 0.3, 0.4], value=0.3, key="rl_noise")
    lams = [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    rows, res = [], {}
    for lam in lams:
        tr, va, acc, wn, g, P, Xtr, ytr = _run(float(lam), int(n), float(noise), 0)
        res[lam] = (g, P, Xtr, ytr); rows.append((f"{lam:g}", f"{tr:.3f}", f"{va:.3f}", f"{va - tr:+.3f}", f"{acc:.3f}", f"{wn:.2f}"))
    table(["λ", "loss", "val_loss", "الفجوة", "val_acc", "‖W‖"], rows, ["num"] * 6)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[np.log10(max(l, 1e-5)) for l in lams], y=[float(r[1]) for r in rows], name="loss", line=dict(color="#2F6FB5", width=3), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=[np.log10(max(l, 1e-5)) for l in lams], y=[float(r[2]) for r in rows], name="val_loss", line=dict(color="#C8473A", width=3), mode="lines+markers"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="log10(λ)  (leftmost = 0)", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="rl_fig")
    intuition("حرف U لخسارة التحقق: λ صغيرة جدًا = فرط تخصيص، كبيرة جدًا = قصور. ‖W‖ تنكمش باطراد مع λ.")
    lam_show = st.select_slider("اعرض حد القرار عند λ", options=lams, value=1e-3, key="rl_show")
    g, P, Xtr, ytr = res[lam_show]
    f2 = go.Figure(go.Contour(x=g, y=g, z=P, colorscale=[[0, "#E6F1FB"], [0.5, "#FFFDF9"], [1, "#FBE6E2"]], contours=dict(start=0, end=1, size=0.1), showscale=False, opacity=0.85))
    f2.add_trace(go.Scatter(x=Xtr[ytr == 1, 0], y=Xtr[ytr == 1, 1], mode="markers", marker=dict(color="#C8473A", size=6), name="y=1"))
    f2.add_trace(go.Scatter(x=Xtr[ytr == 0, 0], y=Xtr[ytr == 0, 1], mode="markers", marker=dict(color="#2F6FB5", size=6), name="y=0"))
    f2.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(scaleanchor="x"), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(f2, width="stretch", key="rl_boundary")
    warning_note("λ = 0.1 قد يجعل الحد شبه خطي (قصور): التنظيم المفرط يقتل القدرة. القاع الأفضل هنا عادةً 1e-3 إلى 1e-2 — ويتغير مع n والضوضاء.")
