"""Learning Rate Lab — LR sweep on a real tiny network, with the five symptoms."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, warning_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.tinynet import TinyNet, make_moons, train

LAB = Lab(
    id="labs.learning_rate_lab",
    title_ar="معمل معدل التعلم",
    title_en="Learning Rate Lab",
    category="training",
    description_ar="مسح لوغاريتمي لمعدل التعلم على شبكة حقيقية: شاهد منحنيات الخسارة لعدة قيم في رسم واحد، وقراءة «حرف U» لخسارة الحقبة الأخيرة مقابل log(η).",
    related_lessons=["foundations.optim.learning_rate", "foundations.optim.schedules_convergence"],
)


@st.cache_data(max_entries=64, show_spinner=False)
def _run(lr: float, epochs: int, optimizer: str, seed: int):
    X, y = make_moons(400, seed=0)
    net = TinyNet([2, 16, 1], "binary", seed=seed)
    return train(net, X[:300], y[:300], X_val=X[300:], y_val=y[300:], epochs=epochs, batch_size=32, lr=lr, optimizer=optimizer, record_grad=True, seed=seed)


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    c1, c2, c3 = st.columns(3)
    with c1:
        optimizer = st.selectbox("المحسّن", ["sgd", "momentum", "adam"], key="lrl_opt")
    with c2:
        epochs = st.slider("الحقب", 5, 40, 15, 5, key="lrl_epochs")
    with c3:
        seed = st.number_input("seed", 0, 99, 0, key="lrl_seed")
    grid = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0] if optimizer != "adam" else [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    chosen = st.multiselect("قيم η", grid, default=grid[1::2], key="lrl_grid")
    intuition("اقرأ المنحنيات كأعراض: مستوٍ (صغير)، ناعم هابط (جيد)، مسنّن (كبير)، منفجر/nan (كبير جدًا).")
    fig = go.Figure(); rows = []; final = []
    palette = ["#2F6FB5", "#1F7A78", "#2E8B57", "#C77A1A", "#C8473A", "#7C5CBF", "#B8860B", "#6B675F", "#000000"]
    for i, lr in enumerate(sorted(chosen)):
        h = _run(float(lr), int(epochs), optimizer, int(seed))
        losses = np.array(h["loss"], dtype=float)
        fig.add_trace(go.Scatter(y=losses, x=np.arange(1, len(losses) + 1), name=f"η={lr}", line=dict(color=palette[i % len(palette)], width=2.5)))
        last = losses[-1] if np.isfinite(losses[-1]) else np.nan
        osc = int((np.diff(losses) > 0).sum()) if np.all(np.isfinite(losses)) else -1
        if not np.isfinite(last):
            verdict = "🔥 تباعد / NaN"
        elif losses[0] - last < 0.02:
            verdict = "🐢 صغير جدًا (بطيء)"
        elif osc >= len(losses) // 3:
            verdict = "〰️ كبير (تذبذب)"
        else:
            verdict = "✅ معقول"
        rows.append((str(lr), f"{last:.4f}" if np.isfinite(last) else "nan", str(osc) if osc >= 0 else "—", f"{np.nanmean(h['grad_norm']):.2f}", verdict))
        final.append((lr, last))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", yaxis=dict(title="train loss", range=[0, 1.5]), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="lrl_fig")
    table(["η", "الخسارة النهائية", "عدد الارتفاعات بين الحقب", "متوسط ‖∇‖", "التشخيص"], rows, ["num", "num", "num", "num", "rtl"])
    if len(final) >= 3:
        f2 = go.Figure(go.Scatter(x=[np.log10(l) for l, _ in final], y=[v if np.isfinite(v) else 1.5 for _, v in final], mode="lines+markers", line=dict(color="#C8473A", width=3)))
        f2.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="خسارة الحقبة الأخيرة مقابل log10(η) — ابحث عن قاع حرف U", xaxis_title="log10(η)", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9")
        st.plotly_chart(f2, width="stretch", key="lrl_u")
    warning_note("الترتيب المعتاد للنطاقات: SGD 0.01–0.5، Momentum أقل قليلًا، Adam 1e-4–1e-2. القاع يتغير مع الشبكة والبيانات وحجم الدفعة — أعد المسح عند تغييرها.")
