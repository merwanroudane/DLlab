"""Optimizer Race — SGD / Momentum / Nesterov / RMSprop / Adam on the same
network, same data, same seed; each with its own tuned learning rate."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, practical_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.tinynet import TinyNet, make_moons, train

LAB = Lab(
    id="labs.optimizer_race",
    title_ar="سباق المحسّنات",
    title_en="Optimizer Race",
    category="training",
    description_ar="خمسة محسّنات على نفس الشبكة والبيانات والبذرة، كل منها بمعدل تعلم تختاره له، ومقارنة خسارة التدريب والتحقق ودقة التحقق حقبةً حقبة.",
    related_lessons=["foundations.optim.optimizer_comparison", "foundations.optim.rmsprop_adam", "foundations.optim.momentum_nesterov"],
)

DEFAULT_LR = {"sgd": 0.3, "momentum": 0.1, "nesterov": 0.1, "rmsprop": 0.01, "adam": 0.01}
COLORS = {"sgd": "#6B675F", "momentum": "#2F6FB5", "nesterov": "#1F7A78", "rmsprop": "#C77A1A", "adam": "#C8473A"}


@st.cache_data(max_entries=64, show_spinner=False)
def _run(opt: str, lr: float, epochs: int, noise: float, seed: int, width: int):
    X, y = make_moons(500, noise=noise, seed=0)
    net = TinyNet([2, width, width, 1], "binary", seed=seed)
    return train(net, X[:400], y[:400], X_val=X[400:], y_val=y[400:], epochs=epochs, batch_size=32, lr=lr, optimizer=opt, seed=seed)


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        epochs = st.slider("الحقب", 5, 60, 25, 5, key="race_epochs")
    with c2:
        noise = st.select_slider("ضوضاء البيانات", options=[0.1, 0.2, 0.3, 0.4], value=0.2, key="race_noise")
    with c3:
        width = st.select_slider("عرض الطبقتين", options=[8, 16, 32], value=16, key="race_width")
    with c4:
        seed = st.number_input("seed", 0, 99, 0, key="race_seed")
    st.markdown("**معدل التعلم لكل محسّن (اضبطه لكل واحد — هذه هي المقارنة العادلة):**")
    cols = st.columns(5); lrs = {}
    for i, opt in enumerate(DEFAULT_LR):
        with cols[i]:
            lrs[opt] = st.select_slider(opt, options=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0], value=DEFAULT_LR[opt], key=f"race_lr_{opt}")
    intuition("الرسم الأيسر خسارة التدريب، الأيمن دقة التحقق. الفائز في التدريب ليس بالضرورة الفائز في التحقق.")
    f1 = go.Figure(); f2 = go.Figure(); rows = []
    for opt, lr in lrs.items():
        h = _run(opt, float(lr), int(epochs), float(noise), int(seed), int(width))
        ep = np.arange(1, len(h["loss"]) + 1)
        f1.add_trace(go.Scatter(x=ep, y=h["loss"], name=opt, line=dict(color=COLORS[opt], width=2.5)))
        f2.add_trace(go.Scatter(x=ep, y=h["val_metric"], name=opt, line=dict(color=COLORS[opt], width=2.5)))
        best_ep = int(np.argmin(h["val_loss"])) + 1 if h["val_loss"] else 0
        rows.append((opt, str(lr), f"{h['loss'][-1]:.4f}" if np.isfinite(h["loss"][-1]) else "nan", f"{h['val_loss'][-1]:.4f}" if h["val_loss"] else "—", f"{h['val_metric'][-1]:.3f}" if h["val_metric"] else "—", str(best_ep)))
    f1.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title="train loss", xaxis_title="epoch", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    f2.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title="validation accuracy", xaxis_title="epoch", yaxis=dict(range=[0.4, 1.0]), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(f1, width="stretch", key="race_f1")
    with c2:
        st.plotly_chart(f2, width="stretch", key="race_f2")
    table(["المحسّن", "η", "خسارة التدريب", "خسارة التحقق", "دقة التحقق", "أفضل حقبة (val_loss)"], rows, ["ltr", "num", "num", "num", "num", "num"])
    practical_note("غيّر البذرة ثلاث مرات قبل إعلان فائز؛ الفروق الصغيرة (< 1%) غالبًا ضوضاء. ولاحظ أن Adam وRMSprop يحتاجان η أصغر بمرتبة من SGD.")
