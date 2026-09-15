"""Overfitting Lab — make a network overfit on purpose, then fix it."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, warning_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.tinynet import TinyNet, make_moons, train

LAB = Lab(
    id="labs.overfitting_lab",
    title_ar="معمل فرط التخصيص",
    title_en="Overfitting Lab",
    category="evaluation",
    description_ar="اصنع فرط تخصيص عمدًا (شبكة كبيرة، بيانات قليلة، ضوضاء، حقب كثيرة)، شاهد الفجوة وحد القرار المتعرج، ثم أصلحه بالإيقاف المبكر أو L2 أو Dropout أو بيانات أكثر.",
    related_lessons=["foundations.generalization.under_overfitting", "foundations.generalization.complexity_data_size"],
)


@st.cache_data(max_entries=32, show_spinner=False)
def _run(n: int, noise: float, width: int, depth: int, epochs: int, l2: float, dropout: float, early: bool, seed: int):
    X, y = make_moons(n + 300, noise=noise, seed=seed)
    Xtr, ytr, Xva, yva = X[:n], y[:n], X[n:], y[n:]
    net = TinyNet([2] + [width] * depth + [1], "binary", seed=seed)
    h = train(net, Xtr, ytr, X_val=Xva, y_val=yva, epochs=epochs, batch_size=16, lr=0.02, optimizer="adam", l2=l2, dropout=dropout, seed=seed, early_stopping_patience=10 if early else None)
    g = np.linspace(-2, 3, 80); GX, GY = np.meshgrid(g, g)
    P = net.predict(np.column_stack([GX.ravel(), GY.ravel()]).astype(np.float32)).reshape(GX.shape)
    return h, Xtr, ytr, g, P, net.n_params()


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    st.markdown("**أسباب الفرط (اجعلها قصوى أولًا):**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n = st.select_slider("حجم التدريب n", options=[30, 60, 120, 300, 600], value=60, key="ovf_n")
    with c2:
        noise = st.select_slider("ضوضاء", options=[0.1, 0.2, 0.3, 0.4], value=0.3, key="ovf_noise")
    with c3:
        width = st.select_slider("العرض", options=[8, 32, 64, 128], value=64, key="ovf_w"); depth = st.select_slider("العمق", options=[1, 2, 3], value=2, key="ovf_d")
    with c4:
        epochs = st.select_slider("الحقب", options=[50, 150, 300, 600], value=300, key="ovf_ep")
    st.markdown("**العلاجات (فعّلها واحدًا واحدًا):**")
    c1, c2, c3 = st.columns(3)
    with c1:
        l2 = st.select_slider("L2 λ", options=[0.0, 1e-4, 1e-3, 1e-2], value=0.0, key="ovf_l2")
    with c2:
        dropout = st.select_slider("Dropout", options=[0.0, 0.2, 0.4, 0.6], value=0.0, key="ovf_do")
    with c3:
        early = st.toggle("إيقاف مبكر (صبر 10)", value=False, key="ovf_es")
    h, Xtr, ytr, g, P, nparams = _run(int(n), float(noise), int(width), int(depth), int(epochs), float(l2), float(dropout), bool(early), 0)
    ep = np.arange(1, len(h["loss"]) + 1)
    gap = h["val_loss"][-1] - h["loss"][-1]
    c1, c2 = st.columns(2)
    with c1:
        f = go.Figure()
        f.add_trace(go.Scatter(x=ep, y=h["loss"], name="loss", line=dict(color="#2F6FB5", width=2.5)))
        f.add_trace(go.Scatter(x=ep, y=h["val_loss"], name="val_loss", line=dict(color="#C8473A", width=2.5)))
        if "stopped_epoch" in h:
            f.add_vline(x=h["stopped_epoch"], line=dict(color="#2E8B57", dash="dot"), annotation_text="early stop")
        f.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
        st.plotly_chart(f, width="stretch", key="ovf_curves")
    with c2:
        f2 = go.Figure(go.Contour(x=g, y=g, z=P, colorscale=[[0, "#E6F1FB"], [0.5, "#FFFDF9"], [1, "#FBE6E2"]], contours=dict(start=0, end=1, size=0.1), showscale=False, opacity=0.85))
        f2.add_trace(go.Scatter(x=Xtr[ytr == 1, 0], y=Xtr[ytr == 1, 1], mode="markers", marker=dict(color="#C8473A", size=6), name="y=1"))
        f2.add_trace(go.Scatter(x=Xtr[ytr == 0, 0], y=Xtr[ytr == 0, 1], mode="markers", marker=dict(color="#2F6FB5", size=6), name="y=0"))
        f2.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(scaleanchor="x"), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
        st.plotly_chart(f2, width="stretch", key="ovf_boundary")
    table(["المعلمات", "الملاحظات", "المعلمات/الملاحظة", "loss النهائية", "val_loss النهائية", "الفجوة", "أفضل val_loss (الحقبة)"],
          [(f"{nparams:,}", str(n), f"{nparams / n:.1f}", f"{h['loss'][-1]:.3f}", f"{h['val_loss'][-1]:.3f}", f"{gap:+.3f}", f"{min(h['val_loss']):.3f} ({int(np.argmin(h['val_loss'])) + 1})")], ["num"] * 7)
    if gap > 0.3:
        warning_note("فرط تخصيص واضح: الفجوة كبيرة والحد متعرج يلتف حول نقاط. جرّب: الإيقاف المبكر (مجاني)، ثم L2 = 1e-3 أو Dropout 0.4، ثم n أكبر.")
    elif h["loss"][-1] > 0.45:
        warning_note("قصور تعلم: الشبكة أو الحقب لا تكفي (أو التنظيم مفرط). قلّل L2/Dropout أو زد القدرة.")
    else:
        st.success("توازن معقول: الفجوة صغيرة والحد ناعم.", icon="✅")
    intuition("لاحظ عمود «المعلمات/الملاحظة»: قيم > 10 تجعل الحفظ سهلًا. البيانات الأكثر هي العلاج الأقوى؛ التنظيم بديل عندما لا تتوفر.")
