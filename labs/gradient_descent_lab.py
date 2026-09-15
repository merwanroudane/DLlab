"""Gradient Descent Lab — batch / mini-batch / SGD on a real tiny network."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, practical_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.tinynet import TinyNet, make_moons, train

LAB = Lab(
    id="labs.gradient_descent_lab",
    title_ar="معمل الانحدار التدريجي",
    title_en="Gradient Descent Lab",
    category="training",
    description_ar="درّب نفس الشبكة الصغيرة بثلاثة أحجام دفعات (كل البيانات، دفعات صغيرة، ملاحظة واحدة) وقارن منحنيات الخسارة بعدد التحديثات، وضوضاء التدرج، ووقت الحقبة.",
    related_lessons=["foundations.optim.gd_variants"],
)


@st.cache_data(max_entries=16, show_spinner=False)
def _run(bs: int, lr: float, epochs: int, seed: int):
    X, y = make_moons(400, seed=0)
    net = TinyNet([2, 16, 1], "binary", seed=seed)
    h = train(net, X[:300], y[:300], X_val=X[300:], y_val=y[300:], epochs=epochs, batch_size=bs, lr=lr, record_grad=True, seed=seed)
    return h


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    c1, c2, c3 = st.columns(3)
    with c1:
        lr = st.select_slider("η", options=[0.01, 0.03, 0.1, 0.3, 0.5, 1.0], value=0.3, key="gdl_lr")
    with c2:
        epochs = st.slider("الحقب", 5, 60, 20, 5, key="gdl_epochs")
    with c3:
        seed = st.number_input("seed", 0, 99, 0, key="gdl_seed")
    configs = [("Batch GD (300)", 300, "#2F6FB5"), ("Mini-batch (32)", 32, "#1F7A78"), ("SGD (1)", 1, "#C8473A")]
    intuition("نفس η للثلاثة عمدًا لترى الفرق الخام. الرسم الأيمن بالحقب، الأيسر بعدد التحديثات — القصة مختلفة في كل رسم.")
    fig1 = go.Figure(); fig2 = go.Figure(); rows = []
    for name, bs, color in configs:
        h = _run(bs, float(lr), int(epochs), int(seed))
        upd = np.arange(1, len(h["loss"]) + 1) * int(np.ceil(300 / bs))
        fig1.add_trace(go.Scatter(y=h["loss"], x=np.arange(1, len(h["loss"]) + 1), name=name, line=dict(color=color, width=2.5)))
        fig2.add_trace(go.Scatter(y=h["loss"], x=upd, name=name, line=dict(color=color, width=2.5)))
        rows.append((name, str(int(np.ceil(300 / bs)) * epochs), f"{h['loss'][-1]:.4f}" if np.isfinite(h["loss"][-1]) else "nan (تباعد)", f"{h['val_metric'][-1]:.3f}" if h["val_metric"] else "—", f"{np.nanmean(h['grad_norm']):.3f}"))
    fig1.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title="الخسارة مقابل الحقب", xaxis_title="epoch", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    fig2.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title="الخسارة مقابل عدد التحديثات", xaxis_title="updates", xaxis_type="log", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig1, width="stretch", key="gdl_f1")
    with c2:
        st.plotly_chart(fig2, width="stretch", key="gdl_f2")
    table(["الإعداد", "إجمالي التحديثات", "الخسارة النهائية", "دقة التحقق", "متوسط ‖∇‖"], rows, ["ltr", "num", "num", "num", "num"])
    practical_note("بالحقب: SGD يبدو الأفضل (300 تحديث لكل حقبة). بالتحديثات: Batch GD يستغل كل تحديث أفضل لكنه يحتاج تمريرة كاملة لكل واحد. الدفعات الصغيرة توازن — وهي الوحيدة التي تستفيد من GPU.")
