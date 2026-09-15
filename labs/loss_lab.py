"""Loss Function Lab — interactive loss curves for regression and classification."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, warning_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table

LAB = Lab(
    id="labs.loss_lab",
    title_ar="معمل دوال الخسارة",
    title_en="Loss Function Lab",
    category="training",
    description_ar="حرّك التنبؤ وشاهد قيمة كل خسارة وتدرجها لحظيًا: MSE/MAE/Huber للانحدار، BCE للثنائي، CCE للمتعدد — مع مقارنة الأزواج الصحيحة والخاطئة.",
    related_lessons=["foundations.loss.mse_mae_huber", "foundations.loss.bce", "foundations.loss.cce_sparse", "foundations.loss.loss_activation_compatibility"],
)


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    mode = st.segmented_control("المهمة", ["انحدار", "ثنائي", "متعدد الفئات"], default="انحدار", key="ll_mode")
    if mode == "انحدار":
        y = st.slider("الحقيقة y", -10.0, 10.0, 3.0, 0.5, key="ll_y"); pred = st.slider("التنبؤ ŷ", -10.0, 10.0, 6.0, 0.5, key="ll_pred")
        delta = st.slider("δ لـ Huber", 0.5, 5.0, 1.0, 0.5, key="ll_delta")
        e = pred - y
        huber = 0.5 * e ** 2 if abs(e) <= delta else delta * (abs(e) - 0.5 * delta)
        gh = e if abs(e) <= delta else delta * np.sign(e)
        table(["الخسارة", "القيمة", "التدرج dL/dŷ"], [("MSE", f"{e ** 2:.3f}", f"{2 * e:+.3f}"), ("MAE", f"{abs(e):.3f}", f"{np.sign(e):+.0f}"), (f"Huber(δ={delta})", f"{huber:.3f}", f"{gh:+.3f}")], ["ltr", "num", "num"])
        es = np.linspace(-10, 10, 400)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=es, y=es ** 2, name="MSE", line=dict(color="#C8473A", width=3)))
        fig.add_trace(go.Scatter(x=es, y=np.abs(es), name="MAE", line=dict(color="#2F6FB5", width=3)))
        fig.add_trace(go.Scatter(x=es, y=np.where(np.abs(es) <= delta, 0.5 * es ** 2, delta * (np.abs(es) - 0.5 * delta)), name="Huber", line=dict(color="#1F7A78", width=3)))
        fig.add_vline(x=e, line=dict(color="#6B675F", dash="dot"), annotation_text=f"e = {e:+.1f}")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="error ŷ − y", yaxis=dict(range=[0, 30]), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch", key="ll_reg")
        intuition("حرّك ŷ بعيدًا عن y: تدرج MSE يكبر بلا حد (خطوات ضخمة مع الشواذ)، تدرج MAE ثابت ±1، وHuber يتوقف عند ±δ.")
    elif mode == "ثنائي":
        y = st.radio("الحقيقة y", [0, 1], index=1, horizontal=True, key="ll_by")
        z = st.slider("logit z", -8.0, 8.0, 1.0, 0.25, key="ll_z")
        p = 1 / (1 + np.exp(-z))
        bce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        table(["الكمية", "القيمة"], [("p̂ = σ(z)", f"{p:.4f}"), ("BCE", f"{bce:.4f}"), ("dBCE/dz = p̂ − y", f"{p - y:+.4f}"), ("MSE على p̂ (للمقارنة، غير مناسبة)", f"{(p - y) ** 2:.4f}"), ("dMSE/dz = 2(p̂−y)p̂(1−p̂)", f"{2 * (p - y) * p * (1 - p):+.4f}")], ["rtl", "num"])
        zs = np.linspace(-8, 8, 400); ps = 1 / (1 + np.exp(-zs))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=zs, y=-(y * np.log(ps) + (1 - y) * np.log(1 - ps)), name="BCE", line=dict(color="#C8473A", width=3)))
        fig.add_trace(go.Scatter(x=zs, y=(ps - y) ** 2, name="MSE on p̂ (wrong choice)", line=dict(color="#B9B2A6", width=2, dash="dot")))
        fig.add_vline(x=z, line=dict(color="#6B675F", dash="dot"))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="logit z", yaxis=dict(range=[0, 8]), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch", key="ll_bin")
        warning_note("لاحظ MSE على الاحتمال: عند z = −8 مع y = 1 (خطأ فادح) تدرجها ≈ 0 لأن Sigmoid مشبعة — النموذج الخاطئ بثقة **لا يتعلم**. BCE تدرجها p̂ − y ≈ −1: إشارة قوية. لهذا لا نستخدم MSE للتصنيف.")
    else:
        K = 3
        c = st.radio("الفئة الصحيحة", [0, 1, 2], horizontal=True, key="ll_c")
        cols = st.columns(3); z = np.array([cols[i].slider(f"logit z{i}", -5.0, 5.0, [2.0, 0.5, -1.0][i], 0.25, key=f"ll_z{i}") for i in range(3)])
        e = np.exp(z - z.max()); p = e / e.sum()
        y = np.eye(3)[c]
        cce = -np.log(p[c])
        table(["الفئة", "logit", "p̂", "y", "التدرج p̂ − y"], [(str(k), f"{z[k]:+.2f}", f"{p[k]:.3f}", str(int(y[k])), f"{p[k] - y[k]:+.3f}") for k in range(3)], ["num", "num", "num", "num", "num"])
        st.code(f"CCE = -log p̂[{c}] = -log({p[c]:.4f}) = {cce:.4f}     reference ln 3 = {np.log(3):.4f}", language="text")
        bce_wrong = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        st.code(f"WRONG pairing (softmax + BCE) would report: {bce_wrong:.4f}  <- treats each class as independent; cannot reach 0 the same way", language="text")
        intuition("ارفع logit الفئة الصحيحة: p̂ ترتفع وCCE تهبط نحو الصفر. ارفع logit فئة خاطئة: CCE ترتفع رغم أن logit الصحيحة لم يتغير — بسبب المقام.")
