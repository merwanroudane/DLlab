import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation, worked_steps
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.loss.bce",
    title_ar="الإنتروبيا المتقاطعة الثنائية BCE",
    title_en="Binary Cross-Entropy (BCE)",
    module="foundations.loss",
    order=3,
    prerequisites=["foundations.loss.error_loss_cost", "foundations.prob.likelihood", "foundations.ml.logistic_regression"],
    objectives_ar=["صيغة BCE وقراءتها كمفاجأة، وقيمتها المرجعية ln 2.", "الفرق بين BCE على احتمالات وBCE على logits (`from_logits`).", "استخدامها للتصنيف متعدد التسميات وبأوزان الفئات."],
    terms=["cross_entropy", "logarithm", "probability"],
    labs=["labs.loss_lab"],
    difficulty="intermediate",
    summary_ar="BCE = −[y log p + (1−y) log(1−p)]؛ مرجع ln 2 = 0.693؛ from_logits للاستقرار؛ لكل عمود في متعدد التسميات.",
)

CODE = '''import numpy as np
def bce(y, p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

def bce_from_logits(y, z):                     # مستقرة عدديًا: log(1+e^{-|z|}) + max(z,0) - z*y
    return np.mean(np.maximum(z, 0) - z*y + np.log1p(np.exp(-np.abs(z))))

y = np.array([1, 0, 1, 1, 0])
p = np.array([0.9, 0.2, 0.6, 0.3, 0.05])
print("BCE (probs) =", bce(y, p).round(4))
z = np.log(p/(1-p))                               # logit العكسي للتحقق
print("BCE (logits)=", bce_from_logits(y, z).round(4), " (same)")

print("uninformed p=0.5 ->", bce(y, np.full(5, 0.5)).round(4), "= ln 2")
print("confident & right p=0.999 on y=1 ->", bce(np.array([1]), np.array([0.999])).round(4))
print("confident & wrong p=0.001 on y=1 ->", bce(np.array([1]), np.array([0.001])).round(4))

# multilabel: BCE لكل عمود ثم متوسط
Y = np.array([[1, 0, 1], [0, 0, 1]]); P = np.array([[0.8, 0.1, 0.7], [0.2, 0.3, 0.9]])
print("multilabel BCE =", bce(Y, P).round(4))
# أوزان الفئات: تضخيم خسارة الفئة النادرة
w_pos = 4.0
weighted = -np.mean(w_pos*y*np.log(p) + (1-y)*np.log(1-p))
print("class-weighted BCE (w_pos=4) =", weighted.round(4))'''


def render() -> None:
    lesson_header(LESSON)
    h2("الصيغة", "The formula")
    equation(r"\text{BCE} = -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\Big]",
             [("y_i", "الحقيقة: 0 أو 1."), (r"\hat{p}_i", "احتمال الفئة 1 من Sigmoid."), (r"y_i \log \hat p_i", "يفعّل عندما $y = 1$: مفاجأة من $\\hat p$."), (r"(1-y_i)\log(1-\hat p_i)", "يفعّل عندما $y = 0$: مفاجأة من $1 - \\hat p$.")],
             meaning_ar="لكل ملاحظة حد واحد فقط حي: $-\\log$ احتمال الفئة الصحيحة. المتوسط على الدفعة.",
             example_ar="$y = 1$، $\\hat p = 0.9$: $-\\ln 0.9 = 0.105$. $y = 0$، $\\hat p = 0.9$: $-\\ln 0.1 = 2.303$.",
             dl_link_ar="`BinaryCrossentropy` في Keras، `BCELoss`/`BCEWithLogitsLoss` في PyTorch. تدرجها بالنسبة لـ logit هو $\\hat p - y$.", title_ar="BCE")
    worked_steps([("y = (1, 0, 1)، p̂ = (0.9, 0.2, 0.6)", r"\text{حدود حية: } -\ln 0.9,\ -\ln 0.8,\ -\ln 0.6"), ("القيم", r"0.105,\ 0.223,\ 0.511"), ("المتوسط", r"0.280")])
    p = np.linspace(0.001, 0.999, 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p, y=-np.log(p), name="y = 1: −log p̂", line=dict(color="#C8473A", width=3)))
    fig.add_trace(go.Scatter(x=p, y=-np.log(1 - p), name="y = 0: −log(1 − p̂)", line=dict(color="#2F6FB5", width=3)))
    fig.add_hline(y=np.log(2), line=dict(color="#B9B2A6", dash="dot"), annotation_text="ln 2 = 0.693 (p̂ = 0.5)")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="p̂", yaxis=dict(range=[0, 6]), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="bce_fig")
    definition("**القيمة المرجعية**: مصنف جاهل ($\\hat p = 0.5$) يعطي $\\ln 2 = 0.693$. خسارة ابتدائية أعلى بكثير = مشكلة (تهيئة، تحجيم، تنشيط مزدوج). خسارة أقل من ذلك مع نموذج لم يتدرب = تسريب أو خطأ في الهدف.")
    code_lab(CodeLab(
        key="loss_bce", title_ar="BCE على احتمالات وعلى logits، قيم مرجعية، متعدد التسميات وأوزان", code=CODE,
        before=Before(goal_ar="حساب BCE بطريقتين متطابقتين، رؤية القيم المرجعية (0.693، 0.001، 6.9)، ثم متعدد التسميات والأوزان.", stage_ar="الخسارة.",
                      inputs_ar="أهداف ثنائية واحتمالات.", expected_ar="BCE ≈ 0.375 بالطريقتين؛ ln 2 للجاهل؛ 6.9 للواثق الخاطئ؛ متعدد التسميات ومرجّح."),
        explain=[("2-4", "القصّ `clip` يمنع log(0). الأطر تفعل مثله."), ("6-7", "صيغة `from_logits`: تجمع Sigmoid والخسارة في تعبير مستقر لا يحسب `1 − p` صغيرة جدًا. **الأفضل دائمًا**."),
                 ("14-16", "المراجع الثلاثة التي يجب أن تحفظها: 0.693، ≈0، ≈7."), ("19-20", "متعدد التسميات: نفس الدالة على مصفوفة (n, K)؛ كل عمود تسمية مستقلة."), ("22-24", "الوزن يضاعف حد الفئة 1: أداة عدم التوازن (`class_weight` / `pos_weight`).")],
        run=run_printed(CODE),
        after_ar="- الطريقتان تعطيان نفس الرقم لكن `from_logits` لا تفشل عند logits كبيرة.\n- الواثق الخاطئ يكلّف 6.9 مقابل 0.001 للواثق الصحيح: الخسارة غير متماثلة عمدًا.",
    ))
    debugging_note("خسارة BCE تبدأ عند ~0.693 ثم لا تتحرك: النموذج يتنبأ بـ 0.5 للجميع. افحص التحجيم والتدرجات. تبدأ عند 5+: logits ضخمة من تهيئة/تحجيم سيئ. تبدأ سالبة: هدف خارج {0,1}.")
    common_mistake("Sigmoid في الطبقة **و** `from_logits=True`: Sigmoid مزدوجة، الاحتمالات تُضغط في (0.5, 0.73) والخسارة لا تنخفض تحت ~0.5 أبدًا.")
    quiz("loss.bce", [
        Q("y = 0، p̂ = 0.25: BCE للملاحظة…", ["−ln 0.25 = 1.39", "−ln 0.75 = 0.288", "0.25"], 1, "الحد الحي هو 1 − p̂."),
        Q("خسارة ابتدائية 0.693 لمصنف ثنائي تعني…", ["خطأ", "نموذج جاهل يتنبأ 0.5 — طبيعي قبل التدريب", "تسريب"], 1, "ln 2."),
        Q("`from_logits=True` تعني…", ["الطبقة الأخيرة بلا Sigmoid والخسارة تطبقه داخليًا", "استخدم Softmax", "لا تستخدم تنشيطًا أبدًا"], 0, "استقرار."),
        Q("BCE لمتعدد التسميات تُحسب…", ["على عمود واحد", "لكل عمود ثم متوسط", "بعد Softmax"], 1, "تسميات مستقلة."),
    ])
    takeaway("BCE = −log احتمال الفئة الصحيحة في المتوسط؛ ln 2 مرجع؛ from_logits دائمًا؛ للثنائي ولكل عمود في متعدد التسميات.")
    lesson_footer(LESSON, ["حد واحد حي لكل ملاحظة.", "0.693 / ≈0 / ≈7 مراجع.", "لا تنشيط مزدوج."])
