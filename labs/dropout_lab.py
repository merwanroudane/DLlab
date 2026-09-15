"""Dropout Lab — train/eval behaviour, rate sweep, and the inverted gap."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, warning_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.tinynet import TinyNet, make_moons, train

LAB = Lab(
    id="labs.dropout_lab",
    title_ar="معمل Dropout",
    title_en="Dropout Lab",
    category="training",
    description_ar="مسح لمعدل الإسقاط p على شبكة كبيرة ببيانات قليلة، منحنيات التدريب/التحقق، والفرق بين تنبؤات وضع التدريب (عشوائية) ووضع الاستدلال (حتمية) لنفس المدخل.",
    related_lessons=["foundations.regularization.dropout"],
)


@st.cache_data(max_entries=32, show_spinner=False)
def _run(p: float, n: int, seed: int):
    X, y = make_moons(n + 300, noise=0.35, seed=seed)
    net = TinyNet([2, 128, 128, 1], "binary", seed=seed)
    h = train(net, X[:n], y[:n], X_val=X[n:], y_val=y[n:], epochs=200, batch_size=16, lr=0.02, optimizer="adam", dropout=p, seed=seed)
    rng = np.random.default_rng(1)
    x0 = X[:1]
    train_preds = [float(net.forward(x0, dropout=p, rng=rng, train=True)[0][0, 0]) for _ in range(8)] if p > 0 else [float(net.predict(x0)[0, 0])] * 8
    eval_pred = float(net.predict(x0)[0, 0])
    return h, train_preds, eval_pred


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    n = st.select_slider("حجم التدريب n", options=[60, 120, 240], value=120, key="dol_n")
    ps = [0.0, 0.2, 0.4, 0.6]
    rows = []; fig = go.Figure()
    colors = ["#6B675F", "#2F6FB5", "#1F7A78", "#C8473A"]
    for p, c in zip(ps, colors):
        h, tp, ep_ = _run(float(p), int(n), 0)
        e = np.arange(1, len(h["loss"]) + 1)
        fig.add_trace(go.Scatter(x=e, y=h["loss"], name=f"loss p={p}", line=dict(color=c, width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=e, y=h["val_loss"], name=f"val_loss p={p}", line=dict(color=c, width=2.5)))
        rows.append((str(p), f"{h['loss'][-1]:.3f}", f"{h['val_loss'][-1]:.3f}", f"{h['val_metric'][-1]:.3f}", f"{min(h['val_loss']):.3f}"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", yaxis=dict(range=[0, 1.2]), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="dol_fig")
    table(["p", "loss (train mode)", "val_loss (eval mode)", "val_acc", "أفضل val_loss"], rows, ["num"] * 5)
    intuition("مع p > 0 قد تكون loss (منقطة) **أعلى** من val_loss (متصلة) طوال التدريب: خسارة التدريب تُحسب بشبكة مشوَّشة. هذا ليس تسريبًا.")
    st.markdown("### نفس المدخل: وضع التدريب مقابل الاستدلال")
    p_show = st.select_slider("p", options=ps, value=0.4, key="dol_p")
    h, tp, ep_ = _run(float(p_show), int(n), 0)
    st.code(f"training-mode predictions (8 random masks): {np.round(tp, 3).tolist()}\neval-mode prediction (deterministic)     : {ep_:.3f}\nstd across training-mode predictions      : {np.std(tp):.3f}", language="text")
    if p_show > 0:
        warning_note("لو نسيت التبديل إلى وضع الاستدلال لحصلت على تنبؤ مختلف في كل استدعاء. في Keras `predict` يتولى ذلك؛ في PyTorch `model.eval()` مسؤوليتك.")
