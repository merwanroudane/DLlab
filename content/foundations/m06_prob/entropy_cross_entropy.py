import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation, worked_steps
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.prob.entropy_cross_entropy",
    title_ar="الإنتروبيا والإنتروبيا المتقاطعة",
    title_en="Entropy & Cross-Entropy",
    module="foundations.prob",
    order=5,
    prerequisites=["foundations.prob.likelihood"],
    objectives_ar=[
        "تعريف الإنتروبيا كمتوسط المفاجأة، والإنتروبيا المتقاطعة كمتوسط المفاجأة عند استخدام توزيع خاطئ.",
        "حساب الإنتروبيا المتقاطعة لمثال تصنيف متعدد الفئات مع one-hot.",
        "ربط CCE بـ Softmax ورؤية لماذا هي خسارة التصنيف القياسية.",
    ],
    terms=["loss", "target"],
    difficulty="intermediate",
    summary_ar="الإنتروبيا متوسط المفاجأة؛ الإنتروبيا المتقاطعة مفاجأة الحقيقة تحت توزيع النموذج — خسارة التصنيف.",
)

CODE = '''import numpy as np

def entropy(p):
    p = np.asarray(p, dtype=float); p = p[p > 0]
    return -(p * np.log(p)).sum()

print("H(uniform 4) =", entropy([.25, .25, .25, .25]).round(4), " (= ln 4)")
print("H(certain)   =", entropy([1, 0, 0, 0]).round(4))
print("H(skewed)    =", entropy([.7, .1, .1, .1]).round(4))

# الإنتروبيا المتقاطعة لمثال واحد: الهدف one-hot، التنبؤ Softmax
def softmax(z): e = np.exp(z - z.max()); return e / e.sum()
def cross_entropy(y_onehot, p): return -(y_onehot * np.log(np.clip(p, 1e-7, 1))).sum()

y = np.array([0, 0, 1])                 # الفئة الصحيحة = 2
for logits in [[0.5, 0.2, 3.0], [1.0, 1.0, 1.0], [3.0, 0.2, 0.5]]:
    p = softmax(np.array(logits))
    print(f"logits={logits} -> p={p.round(3)}  CE={cross_entropy(y, p):.4f}  = -log p[2] = {-np.log(p[2]):.4f}")

# دفعة: متوسط على الملاحظات (ما تفعله categorical_crossentropy)
Y = np.eye(3)[[2, 0, 1]]                # ثلاث ملاحظات بفئات 2, 0, 1
P = np.array([[.1, .2, .7], [.6, .3, .1], [.2, .2, .6]])
print("batch CCE =", np.mean([cross_entropy(Y[i], P[i]) for i in range(3)]).round(4))
print("sparse form (class ids):", (-np.log(P[np.arange(3), [2, 0, 1]])).mean().round(4))'''


def render() -> None:
    lesson_header(LESSON)
    h2("الإنتروبيا: متوسط المفاجأة", "Entropy = average surprise")
    equation(r"H(p) = -\sum_{k} p_k \log p_k", [("p_k", "احتمال الفئة $k$ في التوزيع الحقيقي."), (r"-\log p_k", "مفاجأة رؤية الفئة $k$ (تعلمناها في درس اللوغاريتم)."), ("H", "متوسط المفاجأة موزونًا بالاحتمالات: مقياس «عدم اليقين».")],
             meaning_ar="توزيع متساوٍ = أقصى إنتروبيا (لا نعرف شيئًا)؛ توزيع أكيد = صفر (لا مفاجأة).",
             example_ar="4 فئات متساوية: $H = \\ln 4 \\approx 1.386$؛ فئة أكيدة: $H = 0$.",
             dl_link_ar="إنتروبيا مخرج Softmax تقيس ثقة النموذج. إنتروبيا عالية = مخرج «مسطح» = النموذج لا يعرف.", title_ar="الإنتروبيا")
    intuition("سؤال «هل السماء زرقاء في الصحراء؟» جوابه شبه أكيد: مفاجأة منخفضة. سؤال «أي وجه من ستة سيظهر؟» مفاجأة عالية. الإنتروبيا تقيس مفاجأة السؤال في المتوسط.")
    h2("الإنتروبيا المتقاطعة", "Cross-entropy")
    equation(r"H(y, \hat{p}) = -\sum_{k} y_k \log \hat{p}_k \quad\xrightarrow{\ y \text{ one-hot}\ }\quad -\log \hat{p}_{\text{correct}}",
             [("y_k", "التوزيع الحقيقي؛ في التصنيف one-hot: 1 للفئة الصحيحة و0 للباقي."), (r"\hat{p}_k", "احتمال النموذج (مخرج Softmax) للفئة $k$."), (r"-\log \hat{p}_{\text{correct}}", "بسبب one-hot يبقى حد واحد: مفاجأة النموذج من الفئة الصحيحة.")],
             meaning_ar="كم يتفاجأ النموذج بالحقيقة. تصغيرها = رفع احتمال الفئة الصحيحة.",
             example_ar="الفئة الصحيحة 2 و$\\hat{p} = (0.1, 0.2, 0.7)$: $-\\ln 0.7 = 0.357$. لو $\\hat{p}_2 = 0.1$: $2.303$.",
             dl_link_ar="`categorical_crossentropy` (one-hot) و`sparse_categorical_crossentropy` (أرقام فئات) يحسبان هذا الرقم نفسه ثم يأخذان متوسط الدفعة. وهي بالضبط سالب لوغاريتم الاحتمال الأرجح للتوزيع الفئوي.", title_ar="الإنتروبيا المتقاطعة")
    worked_steps([("الهدف one-hot", r"y = (0, 0, 1)"), ("مخرج Softmax", r"\hat p = (0.1, 0.2, 0.7)"),
                  ("الحدود", r"-(0\cdot\log 0.1 + 0\cdot\log 0.2 + 1\cdot\log 0.7)"), ("النتيجة", r"= -\log 0.7 \approx 0.357")])
    h3("جرّب", "Explore")
    p_correct = st.slider("احتمال الفئة الصحيحة p̂", 0.01, 1.0, 0.7, 0.01, key="ce_p")
    ps = np.linspace(0.01, 1, 200)
    fig = go.Figure(go.Scatter(x=ps, y=-np.log(ps), line=dict(color="#C8473A", width=3)))
    fig.add_trace(go.Scatter(x=[p_correct], y=[-np.log(p_correct)], mode="markers", marker=dict(size=12, color="#1F7A78")))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="p̂(correct)", yaxis_title="cross-entropy", showlegend=False, plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9")
    st.plotly_chart(fig, width="stretch", key="ce_fig")
    st.code(f"CE = -ln({p_correct:.2f}) = {-np.log(p_correct):.4f}", language="text")
    code_lab(CodeLab(
        key="prob_ce", title_ar="إنتروبيا، Softmax، CCE وSparse CCE", code=CODE,
        before=Before(goal_ar="حساب الإنتروبيا لتوزيعات مختلفة، ثم الإنتروبيا المتقاطعة لمخرجات Softmax، ثم متوسط دفعة بصيغتي one-hot وsparse.", stage_ar="احتمال ← خسارة التصنيف.",
                      inputs_ar="توزيعات صغيرة و`logits`.", expected_ar="H(uniform)=1.386، H(certain)=0؛ CE صغيرة عندما يكون logit الفئة الصحيحة الأكبر؛ الصيغتان تعطيان نفس المتوسط."),
        explain=[("3-9", "الإنتروبيا؛ نستبعد الأصفار لأن $0\\log 0 = 0$ بالاتفاق."), ("12-13", "Softmax مستقر عدديًا (طرح الأقصى)؛ CE مع قصّ لتفادي log(0)."),
                 ("15-18", "ثلاث حالات: واثق وصحيح (CE صغيرة)، مسطح (CE = ln 3)، واثق وخاطئ (CE كبيرة)."), ("21-24", "على الدفعة: متوسط. الصيغة `sparse` تختار احتمال الفئة الصحيحة مباشرة بالفهرس — نفس الرقم بلا one-hot.")],
        run=run_printed(CODE),
        after_ar="- المخرج المسطح (1, 1, 1) يعطي CE = ln 3 ≈ 1.0986: القيمة الابتدائية النموذجية لمصنف من 3 فئات قبل التدريب. لـ 10 فئات: ln 10 ≈ 2.303. إن بدأت خسارتك أعلى بكثير فهناك خطأ.\n- الواثق الخاطئ (3.0 لفئة خاطئة) يُعاقب بـ 2.8: أغلى من الجهل.",
    ))
    common_mistake("تمرير `logits` (أرقام حرة) إلى `categorical_crossentropy` التي تتوقع احتمالات، أو تمرير احتمالات مع `from_logits=True`. الخسارة تصبح بلا معنى ولا رسالة خطأ. تحقق دائمًا من تطابق طبقة الإخراج مع إعداد الخسارة.")
    quiz("prob.ce", [
        Q("إنتروبيا توزيع أكيد (1, 0, 0)…", ["ln 3", "0", "1"], 1, "لا مفاجأة."),
        Q("CE عندما تكون p̂(الصحيحة) = 0.5…", ["0.5", "0.693", "2"], 1, "−ln 0.5.", kind="equation"),
        Q("مصنف 10 فئات قبل التدريب: خسارة ابتدائية معقولة ≈", ["0.1", "2.3", "10"], 1, "ln 10.", kind="curve"),
        Q("الفرق بين CCE وSparse CCE…", ["خسارتان مختلفتان", "نفس الرقم؛ شكل الهدف مختلف (one-hot مقابل أرقام)", "Sparse أدق"], 1, "صيغة الهدف فقط."),
    ])
    takeaway("H = متوسط المفاجأة. CE = مفاجأة الحقيقة تحت توزيع النموذج = −log p̂(correct). خسارة التصنيف الابتدائية ≈ ln(k).")
    lesson_footer(LESSON, ["الإنتروبيا تقيس عدم اليقين.", "CCE = −log p̂(correct) في المتوسط.", "ln(k) نقطة مرجعية لتشخيص بداية التدريب."])
