import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto

LESSON = Lesson(
    id="foundations.eval.roc_auc_threshold",
    title_ar="العتبة ومنحنى ROC وAUC ومنحنى الصحة–الاستدعاء",
    title_en="Threshold, ROC Curve, AUC & Precision–Recall",
    module="foundations.eval",
    order=3,
    prerequisites=["foundations.eval.classification_metrics", "foundations.forward.logits_probability_threshold"],
    objectives_ar=["فهم أن المصنف الاحتمالي يعطي عائلة من القرارات بحسب العتبة.", "بناء ROC (TPR مقابل FPR) وقراءة AUC كاحتمال ترتيب صحيح.", "متى تُفضَّل PR على ROC (عدم توازن شديد)، واختيار العتبة بالكلفة."],
    terms=["probability"],
    labs=["labs.threshold_lab"],
    difficulty="intermediate",
    summary_ar="العتبة تختار نقطة على ROC؛ AUC يقيّم الترتيب بغض النظر عن العتبة؛ مع فئة نادرة انظر إلى PR؛ اختر العتبة بالكلفة على التحقق.",
)


def _scores(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.2).astype(int)
    s = np.where(y == 1, rng.normal(1.2, 1.0, n), rng.normal(-0.6, 1.0, n))
    return y, 1 / (1 + np.exp(-s))


def render() -> None:
    lesson_header(LESSON)
    h2("العتبة", "The threshold")
    definition("المصنف يخرج احتمالًا $\\hat p$؛ **العتبة** $\\tau$ تحوّله إلى قرار: إيجابي إن $\\hat p \\ge \\tau$. كل قيمة لـ $\\tau$ تعطي مصفوفة التباس مختلفة. خفض $\\tau$ يرفع الاستدعاء ويخفض الصحة؛ رفعه العكس.")
    y, p = _scores()
    tau = st.slider("العتبة τ", 0.05, 0.95, 0.5, 0.05, key="roc_tau")
    pred = (p >= tau).astype(int)
    TP = ((pred == 1) & (y == 1)).sum(); FP = ((pred == 1) & (y == 0)).sum(); FN = ((pred == 0) & (y == 1)).sum(); TN = ((pred == 0) & (y == 0)).sum()
    tpr, fpr = TP / (TP + FN), FP / (FP + TN); prec = TP / max(TP + FP, 1)
    st.code(f"τ = {tau:.2f}:  TP={TP} FP={FP} FN={FN} TN={TN}   recall(TPR)={tpr:.3f}  FPR={fpr:.3f}  precision={prec:.3f}", language="text")
    taus = np.linspace(0, 1, 201)
    tprs, fprs, precs = [], [], []
    for t in taus:
        pr = (p >= t).astype(int); tp = ((pr == 1) & (y == 1)).sum(); fp = ((pr == 1) & (y == 0)).sum(); fn = ((pr == 0) & (y == 1)).sum(); tn = ((pr == 0) & (y == 0)).sum()
        tprs.append(tp / (tp + fn)); fprs.append(fp / (fp + tn)); precs.append(tp / max(tp + fp, 1))
    auc = float(-np.trapezoid(tprs, fprs))
    c1, c2 = st.columns(2)
    with c1:
        f = go.Figure(go.Scatter(x=fprs, y=tprs, mode="lines", line=dict(color="#1F7A78", width=3), name="ROC"))
        f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#B9B2A6", dash="dot"), name="عشوائي (AUC 0.5)"))
        f.add_trace(go.Scatter(x=[fpr], y=[tpr], mode="markers", marker=dict(size=13, color="#C8473A"), name=f"τ = {tau:.2f}"))
        f.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title=f"ROC — AUC = {auc:.3f}", xaxis_title="FPR = FP/(FP+TN)", yaxis_title="TPR = recall", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
        st.plotly_chart(f, width="stretch", key="roc_fig")
    with c2:
        f2 = go.Figure(go.Scatter(x=tprs, y=precs, mode="lines", line=dict(color="#7C5CBF", width=3), name="PR"))
        f2.add_hline(y=y.mean(), line=dict(color="#B9B2A6", dash="dot"), annotation_text=f"baseline = {y.mean():.2f}")
        f2.add_trace(go.Scatter(x=[tpr], y=[prec], mode="markers", marker=dict(size=13, color="#C8473A"), name=f"τ = {tau:.2f}"))
        f2.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title="Precision–Recall", xaxis_title="recall", yaxis_title="precision", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
        st.plotly_chart(f2, width="stretch", key="pr_fig")
    intuition("حرّك τ: النقطة الحمراء تنزلق على المنحنيين. المنحنى نفسه لا يتغير — هو خاصية **ترتيب** النموذج للملاحظات، والعتبة مجرد اختيار لنقطة عليه.")
    equation(r"\text{AUC} = P\big(\hat p_{\text{positive}} > \hat p_{\text{negative}}\big)",
             [(r"\text{AUC}", "المساحة تحت ROC = احتمال أن يعطي النموذج لإيجابي عشوائي درجة أعلى من سلبي عشوائي."), ("0.5", "ترتيب عشوائي."), ("1.0", "فصل تام.")],
             meaning_ar="مقياس جودة الترتيب مستقل عن العتبة ومستقل عن نسبة الفئات.",
             example_ar="AUC = 0.85: في 85% من الأزواج (إيجابي، سلبي) يرتّبهما النموذج صحيحًا.",
             dl_link_ar="`AUC()` في مقاييس Keras؛ `roc_auc_score` في scikit-learn. جيد للمقارنة بين نماذج؛ لا يخبرك أي عتبة تستخدم.", title_ar="AUC")
    research_note("مع فئة نادرة جدًا (1%) يبدو ROC ممتازًا حتى لنموذج ضعيف لأن FPR يُقسم على عدد سلبيات ضخم. منحنى **الصحة–الاستدعاء** أصدق هناك: خط أساسه = نسبة الإيجابيات، وينهار بوضوح عندما تكثر الإنذارات الكاذبة.")
    h2("اختيار العتبة", "Choosing the threshold")
    st.markdown("""
- **بالكلفة**: إن كانت كلفة التفويت $c_{FN}$ وكلفة الإنذار الكاذب $c_{FP}$، اختر $\\tau$ التي تقلل $c_{FN}\\cdot FN + c_{FP}\\cdot FP$ على **التحقق**.
- **بمقياس مستهدف**: أعلى F1، أو أعلى استدعاء عند صحة ≥ 0.8، حسب متطلبات القرار.
- **افتراضي 0.5** صحيح فقط عندما تتساوى الكلفتان وتكون الاحتمالات معايَرة.
""")
    st.button("افتح معمل العتبة", icon=":material/science:", type="primary", on_click=goto, args=("labs.threshold_lab",), key="thr_lab")
    common_mistake("اختيار العتبة على مجموعة الاختبار ثم الإبلاغ عن F1 هناك: العتبة معلمة فائقة تُضبط على التحقق كغيرها.")
    quiz("eval.roc", [
        Q("خفض العتبة…", ["يرفع الصحة ويخفض الاستدعاء", "يرفع الاستدعاء ويخفض الصحة", "لا يغيّر شيئًا"], 1, "إيجابيات أكثر."),
        Q("AUC = 0.5 يعني…", ["نموذج مثالي", "ترتيب عشوائي", "خطأ في الكود"], 1, "لا تمييز."),
        Q("فئة نادرة 0.5%: أي منحنى أصدق؟", ["ROC", "Precision–Recall", "متساويان"], 1, "خط أساس PR = 0.005."),
        Q("العتبة تُضبط على…", ["التدريب", "التحقق", "الاختبار"], 1, "معلمة فائقة."),
    ])
    takeaway("العتبة نقطة على المنحنى؛ AUC جودة الترتيب؛ PR للنادر؛ اختر العتبة بالكلفة على التحقق.")
    lesson_footer(LESSON, ["ROC: TPR مقابل FPR.", "AUC = احتمال ترتيب صحيح.", "PR أصدق مع عدم التوازن الشديد."])
