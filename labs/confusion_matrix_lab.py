"""Confusion Matrix Lab — edit the four cells (or class counts) and see every metric."""

import numpy as np
import streamlit as st

from components.callouts import intuition, warning_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table

LAB = Lab(
    id="labs.confusion_matrix_lab",
    title_ar="معمل مصفوفة الالتباس",
    title_en="Confusion Matrix Lab",
    category="evaluation",
    description_ar="عدّل TP/FP/FN/TN مباشرة وشاهد الدقة والصحة والاستدعاء وF1 وخط الأساس تتغير؛ ثم مصفوفة متعددة الفئات مع macro/micro.",
    related_lessons=["foundations.eval.classification_metrics", "foundations.eval.metric_selection"],
)


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    st.markdown("### ثنائي")
    c1, c2 = st.columns(2)
    with c1:
        TP = st.number_input("TP (إيجابي صحيح)", 0, 10000, 40, key="cml_tp"); FN = st.number_input("FN (تفويت)", 0, 10000, 10, key="cml_fn")
    with c2:
        FP = st.number_input("FP (إنذار كاذب)", 0, 10000, 30, key="cml_fp"); TN = st.number_input("TN (سلبي صحيح)", 0, 10000, 920, key="cml_tn")
    n = TP + FP + FN + TN
    if n == 0:
        st.info("أدخل قيمًا.")
        return
    acc = (TP + TN) / n; prec = TP / max(TP + FP, 1); rec = TP / max(TP + FN, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    spec = TN / max(TN + FP, 1); pos_rate = (TP + FN) / n; base = max(pos_rate, 1 - pos_rate)
    table(["", "تنبؤ: إيجابي", "تنبؤ: سلبي"], [("حقيقة: إيجابي", str(TP), str(FN)), ("حقيقة: سلبي", str(FP), str(TN))], ["rtl", "num", "num"])
    table(["المقياس", "القيمة", "الصيغة", "يجيب عن"],
          [("الدقة", f"{acc:.3f}", "(TP+TN)/n", "نسبة الصواب الكلية"), ("خط الأساس (الأغلب)", f"{base:.3f}", "max(p, 1−p)", "ما يحققه نموذج لا يتعلم"),
           ("الصحة", f"{prec:.3f}", "TP/(TP+FP)", "من الإيجابيات المتنبأة كم صحيح"), ("الاستدعاء", f"{rec:.3f}", "TP/(TP+FN)", "من الإيجابيات الحقيقية كم التقطنا"),
           ("النوعية", f"{spec:.3f}", "TN/(TN+FP)", "من السلبيات كم صُنّف صحيحًا"), ("F1", f"{f1:.3f}", "2PR/(P+R)", "توازن الصحة والاستدعاء")],
          ["rtl", "num", "code", "rtl"])
    intuition("جرّب: TP=0, FN=50, FP=0, TN=950 — دقة 95% وF1 = 0: نموذج «سلبي دائمًا». ثم TP=50, FN=0, FP=950, TN=0 — استدعاء 100% وصحة 5%: نموذج «إيجابي دائمًا».")
    if acc <= base + 0.005 and TP + FP > 0:
        warning_note("الدقة لا تتجاوز خط الأساس: النموذج لم يتعلم شيئًا ذا قيمة رغم رقم الدقة.")
    st.markdown("### متعدد الفئات")
    K = st.slider("عدد الفئات", 2, 5, 3, key="cml_k")
    rng = np.random.default_rng(0)
    default = np.eye(K, dtype=int) * 30 + rng.integers(0, 8, (K, K))
    import pandas as pd
    df = pd.DataFrame(default, index=[f"true {i}" for i in range(K)], columns=[f"pred {j}" for j in range(K)])
    edited = st.data_editor(df, key="cml_multi", width="stretch")
    cm = edited.to_numpy(dtype=float)
    prec_k = np.diag(cm) / np.maximum(cm.sum(0), 1); rec_k = np.diag(cm) / np.maximum(cm.sum(1), 1); f1_k = 2 * prec_k * rec_k / np.maximum(prec_k + rec_k, 1e-9)
    rows = [(f"{i}", str(int(cm[i].sum())), f"{prec_k[i]:.3f}", f"{rec_k[i]:.3f}", f"{f1_k[i]:.3f}") for i in range(K)]
    table(["الفئة", "الدعم (n)", "الصحة", "الاستدعاء", "F1"], rows, ["num", "num", "num", "num", "num"])
    st.code(f"accuracy (= micro-F1) = {np.diag(cm).sum() / max(cm.sum(), 1):.3f}    macro-F1 = {f1_k.mean():.3f}    weighted-F1 = {(f1_k * cm.sum(1) / max(cm.sum(), 1)).sum():.3f}", language="text")
    st.caption("عدّل الخلايا خارج القطر لترى أي فئتين تلتبسان وكيف يهبط macro-F1 قبل الدقة عندما تتأثر فئة صغيرة.")
