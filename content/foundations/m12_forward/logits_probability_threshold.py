import numpy as np
import streamlit as st

from components.callouts import common_mistake, definition, practical_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import pipeline

LESSON = Lesson(
    id="foundations.forward.logits_probability_threshold",
    title_ar="التنبؤ: logits، الاحتمال، العتبة، والفئة المتنبأة",
    title_en="Prediction: Logits, Probability, Threshold & Predicted Class",
    module="foundations.forward",
    order=2,
    prerequisites=["foundations.forward.layer_by_layer", "foundations.activations.softmax_output"],
    objectives_ar=["التمييز بين المراحل الثلاث لمخرج المصنف: logits ← احتمال ← قرار.", "اختيار العتبة كقرار كلفة، واستخدام argmax للمتعدد.", "قراءة مخرج `predict()` لكل نوع مهمة."],
    terms=["softmax", "probability", "prediction"],
    difficulty="beginner",
    summary_ar="logits أرقام حرة؛ Sigmoid/Softmax تحوّلها إلى احتمالات؛ العتبة/argmax يحوّلها إلى فئة. predict يعيد الاحتمال غالبًا لا الفئة.",
)

CODE = '''import numpy as np
sigmoid = lambda z: 1/(1+np.exp(-z))
def softmax(z): e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

# ثنائي: logits (n,1) → احتمال → قرار بعتبة
logits_bin = np.array([[2.1], [-0.4], [0.3], [-3.0]])
p = sigmoid(logits_bin)
for thr in [0.5, 0.3]:
    print(f"thr={thr}: probs={p[:,0].round(3)} -> classes={(p[:,0] >= thr).astype(int)}")

# متعدد: logits (n,3) → Softmax → argmax
logits_mc = np.array([[2.0, 0.5, -1.0], [0.1, 0.2, 0.15], [-2.0, 3.0, 2.9]])
probs = softmax(logits_mc)
print("probs:\\n", probs.round(3))
print("argmax:", probs.argmax(axis=1), " confidence:", probs.max(axis=1).round(3))
print("row 1 is nearly uniform: low confidence even though argmax exists")

# انحدار: المخرج هو التنبؤ مباشرة
print("regression output:", np.array([[215000.0], [98000.0]])[:, 0])'''


def render() -> None:
    lesson_header(LESSON)
    h2("ثلاث مراحل", "Three stages")
    pipeline(["last layer z (logits)", "activation → probability", "threshold / argmax → class"], active=0)
    definition("**Logits**: مخرج الطبقة الأخيرة قبل التنشيط؛ أرقام حرة (−∞, ∞). **الاحتمال**: بعد Sigmoid (ثنائي) أو Softmax (متعدد). **الفئة المتنبأة**: بعد عتبة (ثنائي) أو `argmax` (متعدد). في الانحدار لا توجد مراحل: المخرج الخطي هو التنبؤ.")
    compare_table(["المهمة", "شكل logits", "التحويل", "شكل الاحتمال", "القرار", "ما يعيده predict عادةً"],
                  [("ثنائي", "(n, 1)", "Sigmoid", "(n, 1) في (0,1)", "p ≥ thr", "الاحتمال (n, 1)"), ("متعدد (K)", "(n, K)", "Softmax", "(n, K) مجموع كل صف 1", "argmax على المحور 1", "الاحتمالات (n, K)"),
                   ("متعدد التسميات", "(n, K)", "Sigmoid لكل عمود", "(n, K) كل عمود مستقل", "كل عمود ≥ عتبته", "الاحتمالات (n, K)"), ("انحدار", "(n, 1)", "لا شيء", "—", "—", "القيمة (n, 1)")],
                  ["rtl", "code", "ltr", "code", "code", "rtl"])
    code_lab(CodeLab(
        key="fwd_logits", title_ar="من logits إلى قرار في الحالات الثلاث", code=CODE,
        before=Before(goal_ar="تحويل logits إلى احتمالات ثم إلى فئات بعتبتين مختلفتين، وبـ argmax للمتعدد، ورؤية الثقة.", stage_ar="التمرير الأمامي ← التنبؤ.",
                      inputs_ar="logits مصطنعة.", expected_ar="عتبة 0.3 تصنّف ملاحظة إضافية إيجابيًا؛ الصف الثاني متعدد الفئات شبه متساوٍ (ثقة 0.35)."),
        explain=[("6-9", "نفس الاحتمالات، عتبتان: القرار يتغير للملاحظة الثالثة (0.574) — لا، بل للثانية (0.401 ≥ 0.3). العتبة قرار."),
                 ("12-16", "Softmax صفًا صفًا؛ argmax يعطي الفئة، max يعطي الثقة. الصف الثاني: argmax موجود دائمًا حتى لو كانت الثقة 0.35 — لا تبلّغ عن الفئة دون ثقتها."),
                 ("19", "الانحدار: لا تحويل.")],
        run=run_printed(CODE),
        after_ar="- `predict` في الأطر يعيد **الاحتمالات** (أو القيم) لا الفئات؛ القرار مسؤوليتك.\n- أبلغ عن الثقة مع الفئة؛ argmax بثقة 0.35 على 3 فئات يكاد يكون تخمينًا.",
    ))
    practical_note("العتبة قرار اقتصادي: إن كانت كلفة تفويت متعثر (خسارة القرض) أكبر بكثير من كلفة إزعاج عميل جيد، فاخفض العتبة. تُضبط على مجموعة التحقق بمنحنى Precision–Recall (الوحدة 18).")
    common_mistake("تمرير logits إلى `argmax` مباشرة: النتيجة صحيحة (Softmax رتيبة لا تغيّر الترتيب) لكن قراءتها كاحتمالات خاطئة: 2.0 ليس «200%».")
    quiz("fwd.logits", [
        Q("logits (n, 5) لتصنيف 5 فئات؛ للحصول على الفئة…", ["np.argmax(logits, axis=1)", "logits ≥ 0.5", "np.sum(logits)"], 0, "argmax على محور الفئات.", kind="code"),
        Q("`model.predict` لمصنف ثنائي يعيد…", ["0 أو 1", "احتمالًا في (0,1)", "logits دائمًا"], 1, "القرار عليك."),
        Q("Softmax تغيّر ترتيب الفئات؟", ["نعم", "لا؛ رتيبة", "أحيانًا"], 1, "argmax على logits = argmax على الاحتمالات."),
    ])
    takeaway("logits ← احتمال ← قرار. العتبة كلفة، argmax للمتعدد، الثقة تُبلَّغ مع الفئة. predict يعيد الاحتمال.")
    lesson_footer(LESSON, ["ثلاث مراحل بثلاثة أشكال.", "العتبة تُضبط على التحقق.", "argmax لا يعني ثقة."])
