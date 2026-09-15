import numpy as np
import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, intuition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation, worked_steps
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.activations.softmax_output",
    title_ar="Softmax وتنشيط طبقة الإخراج حسب المهمة",
    title_en="Softmax & Output-layer Activation by Task",
    module="foundations.activations",
    order=5,
    prerequisites=["foundations.activations.sigmoid_tanh", "foundations.prob.entropy_cross_entropy"],
    objectives_ar=["حساب Softmax يدويًا وفهم أنها تعمل على متجه كامل.", "فهم استقرارها العددي (طرح الأقصى) وحساسيتها للمقياس (درجة الحرارة).", "ملء جدول: مهمة ← تنشيط إخراج ← خسارة."],
    terms=["softmax", "cross_entropy", "probability"],
    difficulty="intermediate",
    summary_ar="Softmax = exp(z_k)/Σexp(z_j): احتمالات مجموعها 1 من logits؛ تُقرن بـ CCE. الإخراج: خطي/Sigmoid/Softmax حسب المهمة.",
)

CODE = '''import numpy as np
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))     # طرح الأقصى: استقرار عددي بلا تغيير النتيجة
    return e / e.sum(axis=-1, keepdims=True)

z = np.array([2.0, 1.0, 0.1])
p = softmax(z)
print("logits:", z, " probs:", p.round(4), " sum:", p.sum().round(6), " argmax:", p.argmax())

# الحساسية للمقياس (درجة الحرارة T): z/T
for T in [0.5, 1, 2, 10]:
    print(f"T={T:>4}: {softmax(z / T).round(3)}")

# الاستقرار: logits كبيرة تنفجر بدون طرح الأقصى
big = np.array([1000.0, 999.0, 998.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(big) / np.exp(big).sum()
print("naive:", naive, "  stable:", softmax(big).round(4))

# دفعة: Softmax لكل صف مستقلًا
Z = np.array([[2.0, 1.0, 0.1], [0.0, 0.0, 0.0], [-1.0, 3.0, -1.0]])
print(softmax(Z).round(3), softmax(Z).sum(axis=1))'''


def render() -> None:
    lesson_header(LESSON)
    h2("Softmax", "Softmax")
    equation(r"\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}",
             [(r"\mathbf{z}", "متجه `logits` بطول $K$ (عدد الفئات): مخرجات خطية حرة."), ("e^{z_k}", "يجعل كل قيمة موجبة (درس الأس)."), (r"\sum_j e^{z_j}", "القسمة على المجموع تجعل المجموع 1.")],
             meaning_ar="تحويل $K$ أرقام حرة إلى توزيع احتمالي. الأكبر يحصل على أكبر احتمال؛ الفروق تُضخَّم أسّيًا.",
             example_ar="$z = (2, 1, 0.1)$ ⟹ $p \\approx (0.66, 0.24, 0.10)$.",
             dl_link_ar="طبقة الإخراج للتصنيف متعدد الفئات `Dense(K, activation='softmax')` مع `categorical_crossentropy`. تدرج (Softmax + CCE) بالنسبة لـ z هو ببساطة $p - y$ — نفس أناقة Sigmoid + BCE.", title_ar="Softmax")
    worked_steps([("الأسية", r"e^{2} = 7.39,\ e^{1} = 2.72,\ e^{0.1} = 1.11"), ("المجموع", r"7.39 + 2.72 + 1.11 = 11.21"), ("القسمة", r"p = (0.659,\ 0.242,\ 0.099)")])
    intuition("Softmax «انتخاب مرجّح»: كل فئة تحصل على أصوات بقدر $e^{z}$، والاحتمال هو نصيبها من المجموع. مضاعفة كل logits (درجة حرارة منخفضة) تجعل الفائز يكتسح؛ تصغيرها تجعل التوزيع مسطحًا.")
    code_lab(CodeLab(
        key="act_softmax", title_ar="Softmax مستقرة، درجة الحرارة، والدفعة", code=CODE,
        before=Before(goal_ar="حساب Softmax بصيغة مستقرة عدديًا، ورؤية أثر المقياس، وفشل الصيغة الساذجة مع logits كبيرة، وتطبيقها صفًا صفًا على دفعة.", stage_ar="التنشيط ← الإخراج.",
                      inputs_ar="متجه logits، ودفعة 3×3.", expected_ar="احتمالات مجموعها 1؛ T صغيرة تحدّد، كبيرة تسطّح؛ الساذجة تعطي NaN والمستقرة تعمل؛ كل صف مجموعه 1."),
        explain=[("2-4", "طرح الأقصى لا يغيّر النتيجة رياضيًا (يُختصر في البسط والمقام) لكنه يمنع `exp(1000) = inf`. الأطر تفعل هذا داخليًا."),
                 ("10-12", "درجة الحرارة: قسمة logits على T. T→0 يعطي one-hot تقريبًا، T→∞ يعطي توزيعًا متساويًا."),
                 ("15-18", "`inf / inf = nan`: سبب كلاسيكي لخسارة NaN عند تمرير logits ضخمة إلى Softmax يدوية."), ("21-22", "`axis=-1`: Softmax على محور الفئات لكل ملاحظة مستقلة.")],
        run=run_printed(CODE),
        after_ar="- `argmax` لا يتأثر بـ T: الفئة المتنبأة نفسها، الثقة تختلف.\n- استخدم دائمًا Softmax الإطار أو `from_logits=True` في الخسارة بدل تنفيذ ساذج.",
    ))
    h2("جدول الإخراج النهائي", "The output table")
    compare_table(["المهمة", "الوحدات", "التنشيط", "الخسارة", "التنبؤ النهائي"],
                  [("انحدار", "1 (أو عدد المخرجات)", "خطي", "MSE / MAE / Huber", "القيمة مباشرة"),
                   ("ثنائي", "1", "Sigmoid", "BCE", "p ≥ عتبة"), ("متعدد الفئات", "K", "Softmax", "CCE / Sparse CCE", "argmax"),
                   ("متعدد التسميات", "K", "Sigmoid لكل وحدة", "BCE", "كل وحدة ≥ عتبتها"), ("انحدار موجب (عدّ، سعر)", "1", "خطي (أو softplus/exp)", "MSE على log أو Poisson", "القيمة")],
                  ["rtl", "code", "ltr", "ltr", "rtl"])
    debugging_note("`from_logits=True` في خسائر Keras تعني: «لا تنشيط في الإخراج، الخسارة تطبق Softmax/Sigmoid داخليًا بصورة مستقرة». إن وضعت Softmax في الطبقة **و** `from_logits=True` تُطبَّق مرتين — خسارة بلا معنى بلا رسالة خطأ.")
    common_mistake("Softmax للتصنيف الثنائي بوحدتين + Sigmoid لكل وحدة معًا، أو Softmax مع BCE. اختر صفًا واحدًا من الجدول.")
    quiz("act.softmax", [
        Q("logits (3, 3, 3): Softmax تعطي…", ["(1, 0, 0)", "(1/3, 1/3, 1/3)", "(3, 3, 3)"], 1, "متساوية."),
        Q("لماذا نطرح الأقصى قبل exp؟", ["لتغيير النتيجة", "للاستقرار العددي دون تغيير النتيجة", "لتسريع الحساب"], 1, "منع inf."),
        Q("Softmax في الطبقة + from_logits=True…", ["صحيح", "تنشيط مزدوج: خطأ صامت", "أسرع"], 1, "مرتين."),
        Q("تصنيف 4 فئات: طبقة الإخراج…", ["Dense(1, sigmoid)", "Dense(4, softmax)", "Dense(4, relu)"], 1, "الصف الثالث."),
    ])
    takeaway("Softmax تحوّل logits إلى توزيع؛ اطرح الأقصى؛ اقرنها بـ CCE. جدول المهمة ← التنشيط ← الخسارة هو مرجعك الدائم.")
    lesson_footer(LESSON, ["exp/Σexp على محور الفئات.", "درجة الحرارة تغيّر الثقة لا القرار.", "from_logits يعني بلا تنشيط في الطبقة."])
