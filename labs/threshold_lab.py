"""Threshold Lab — choose the decision threshold by cost on real model scores."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, practical_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.datasets import loan_default

LAB = Lab(
    id="labs.threshold_lab",
    title_ar="معمل العتبة",
    title_en="Threshold Lab",
    category="evaluation",
    description_ar="درجات احتمال حقيقية من نموذج لوجستي على بيانات القروض: حرّك العتبة، حدد كلفة التفويت وكلفة الإنذار الكاذب، وشاهد الكلفة الكلية والصحة والاستدعاء، والعتبة المثلى.",
    related_lessons=["foundations.eval.roc_auc_threshold", "foundations.eval.metric_selection"],
)


@st.cache_data(show_spinner=False)
def _scores():
    df = loan_default().dropna()
    X = df[["income", "age", "num_late_payments", "debt_ratio"]].to_numpy(float); y = df["defaulted"].to_numpy(float)
    rng = np.random.default_rng(0); idx = rng.permutation(len(X)); tr, va = idx[:280], idx[280:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Z = (X - mu) / sd
    Zb = np.column_stack([Z, np.ones(len(Z))]); w = np.zeros(Zb.shape[1])
    for _ in range(400):
        p = 1 / (1 + np.exp(-(Zb[tr] @ w))); w -= 0.5 * Zb[tr].T @ (p - y[tr]) / len(tr)
    p_va = 1 / (1 + np.exp(-(Zb[va] @ w)))
    return p_va, y[va]


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    p, y = _scores()
    c1, c2, c3 = st.columns(3)
    with c1:
        tau = st.slider("العتبة τ", 0.02, 0.98, 0.5, 0.02, key="thl_tau")
    with c2:
        c_fn = st.number_input("كلفة التفويت (FN) — قرض متعثر لم نكتشفه", 0, 100000, 5000, 500, key="thl_cfn")
    with c3:
        c_fp = st.number_input("كلفة الإنذار الكاذب (FP) — عميل جيد رُفض", 0, 100000, 500, 100, key="thl_cfp")
    taus = np.linspace(0.01, 0.99, 197)
    def stats(t):
        pr = (p >= t); TP = (pr & (y == 1)).sum(); FP = (pr & (y == 0)).sum(); FN = ((~pr) & (y == 1)).sum(); TN = ((~pr) & (y == 0)).sum()
        return TP, FP, FN, TN
    costs = [c_fn * stats(t)[2] + c_fp * stats(t)[1] for t in taus]
    best_t = float(taus[int(np.argmin(costs))])
    TP, FP, FN, TN = stats(tau)
    prec = TP / max(TP + FP, 1); rec = TP / max(TP + FN, 1); cost = c_fn * FN + c_fp * FP
    table(["عند τ", "TP", "FP", "FN", "TN", "الصحة", "الاستدعاء", "الكلفة الكلية"],
          [(f"{tau:.2f}", str(TP), str(FP), str(FN), str(TN), f"{prec:.3f}", f"{rec:.3f}", f"{cost:,}")], ["num"] * 8)
    st.code(f"optimal threshold for these costs = {best_t:.2f}   (min cost = {min(costs):,.0f})\nbase rate of default in validation = {y.mean():.3f}", language="text")
    fig = go.Figure(go.Scatter(x=taus, y=costs, line=dict(color="#C8473A", width=3), name="الكلفة الكلية"))
    fig.add_vline(x=tau, line=dict(color="#1F7A78", dash="dot"), annotation_text=f"τ = {tau:.2f}")
    fig.add_vline(x=best_t, line=dict(color="#2E8B57"), annotation_text=f"optimal {best_t:.2f}")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="threshold", yaxis_title="expected cost", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9")
    st.plotly_chart(fig, width="stretch", key="thl_fig")
    hist = go.Figure()
    hist.add_trace(go.Histogram(x=p[y == 0], nbinsx=30, name="y = 0", marker_color="#2F6FB5", opacity=0.6))
    hist.add_trace(go.Histogram(x=p[y == 1], nbinsx=30, name="y = 1", marker_color="#C8473A", opacity=0.6))
    hist.add_vline(x=tau, line=dict(color="#1F7A78", dash="dot"))
    hist.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10), barmode="overlay", xaxis_title="predicted probability", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(hist, width="stretch", key="thl_hist")
    intuition("التوزيعان يتداخلان: لا عتبة تفصلهما تمامًا. رفع كلفة التفويت يدفع العتبة المثلى إلى اليسار (نُصنّف إيجابيًا أكثر)؛ رفع كلفة الإنذار الكاذب يدفعها إلى اليمين.")
    practical_note("هذه العتبة ضُبطت على مجموعة **تحقق**. في التقرير النهائي طبّقها كما هي على الاختبار ولا تعد ضبطها هناك.")
