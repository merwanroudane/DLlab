import numpy as np
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation, worked_steps
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.linalg.dot_product",
    title_ar="الضرب النقطي",
    title_en="Dot Product",
    module="foundations.linalg",
    order=2,
    prerequisites=["foundations.linalg.scalars_vectors"],
    objectives_ar=[
        "حساب الضرب النقطي كمجموع حواصل ضرب المكوّنات.",
        "فهم معناه: مقياس توافق اتجاهين، وأن الخلية العصبية ضرب نقطي + انحياز.",
        "استخدام `np.dot` و`@` والتمييز عن `*`.",
    ],
    terms=["weight", "bias"],
    labs=["labs.vector_matrix"],
    difficulty="beginner",
    summary_ar="w·x = Σ wᵢxᵢ: رقم واحد يقيس توافق الوزن مع المدخل — قلب الخلية العصبية.",
)

CODE = '''import numpy as np

x = np.array([2.0, 0.0, 1.0])       # ملاحظة: 3 خصائص
w = np.array([0.5, -1.0, 2.0])      # أوزان الخلية
b = 0.1

print((w * x))                      # ضرب عنصري: متجه (ليس الضرب النقطي!)
print((w * x).sum())                # مجموعه = الضرب النقطي
print(np.dot(w, x), w @ x)          # الطريقتان القياسيتان
z = w @ x + b                       # الخلية العصبية قبل التنشيط
print("z =", z)

# التوافق: نفس الاتجاه، متعامد، معاكس
a = np.array([1.0, 0.0])
print(a @ np.array([1.0, 0.0]), a @ np.array([0.0, 1.0]), a @ np.array([-1.0, 0.0]))

# دفعة من 4 ملاحظات: ضرب نقطي لكل صف في سطر واحد
X = np.array([[2, 0, 1], [1, 1, 1], [0, 3, 0], [4, 0, 0]], dtype=float)
print(X @ w + b)                    # (4,3)@(3,) → (4,)'''


def render() -> None:
    lesson_header(LESSON)
    h2("التعريف", "Definition")
    equation(r"\mathbf{w}\cdot\mathbf{x} = \sum_{j=1}^{n} w_j x_j = w_1x_1 + w_2x_2 + \cdots + w_nx_n",
             [(r"\mathbf{w}\cdot\mathbf{x}", "الضرب النقطي: **عدد واحد** من متجهين بنفس الطول."), ("w_j x_j", "حاصل ضرب المكوّنتين المتقابلتين."), (r"\sum", "ثم الجمع.")],
             meaning_ar="اضرب كل مكوّنة في مقابلتها واجمع.",
             example_ar="$(0.5, -1, 2)\\cdot(2, 0, 1) = 1 + 0 + 2 = 3$.",
             dl_link_ar="الخلية العصبية: $z = \\mathbf{w}\\cdot\\mathbf{x} + b$. الأوزان تقول «كم أهتم بكل خاصية»، والضرب النقطي يلخص المدخل في رقم واحد.", title_ar="الضرب النقطي")
    worked_steps([("المكوّنات المتقابلة", r"(0.5)(2),\ (-1)(0),\ (2)(1)"), ("الحواصل", r"1,\ 0,\ 2"), ("المجموع", r"1 + 0 + 2 = 3"), ("مع الانحياز", r"z = 3 + 0.1 = 3.1")])
    h2("المعنى", "Meaning")
    definition("هندسيًا: $\\mathbf{w}\\cdot\\mathbf{x} = \\|\\mathbf{w}\\|\\,\\|\\mathbf{x}\\|\\cos\\theta$. موجب كبير عندما يتجه المتجهان نفس الاتجاه، صفر عندما يتعامدان، سالب عندما يتعاكسان.")
    intuition("الأوزان «قالب»؛ الضرب النقطي يقيس كم يشبه المدخل القالب. خلية تعلمت أوزانًا موجبة للدخل وسالبة للتأخيرات تعطي $z$ كبيرًا للعميل الجيد وصغيرًا للمتعثر.")
    why("لماذا لا نكتفي بالضرب العنصري `w * x`؟ لأنه يعطي متجهًا لا قرارًا. الجمع هو ما يحوّل $n$ أدلة إلى حكم واحد.")
    code_lab(CodeLab(
        key="la_dot", title_ar="الضرب النقطي = الخلية العصبية", code=CODE,
        before=Before(goal_ar="حساب الضرب النقطي بثلاث طرق، تفسير إشارته، وتطبيقه على دفعة كاملة.", stage_ar="جبر خطي ← الخلية.",
                      inputs_ar="متجه أوزان `(3,)`، ملاحظة `(3,)`، ودفعة `(4, 3)`.", expected_ar="`w * x` متجه، مجموعه 3.0، `z = 3.1`، ثلاث قيم توافق (1, 0, −1)، وأربع قيم `z` للدفعة.",
                      math_ar="$z = \\mathbf{w}\\cdot\\mathbf{x} + b$ ثم $\\mathbf{z} = X\\mathbf{w} + b$ للدفعة."),
        explain=[("7-8", "`*` عنصري يعطي `(3,)`؛ جمعه هو الضرب النقطي. الفرق بين `*` و`@` أهم خطأ في هذه الوحدة."),
                 ("9-11", "`np.dot` أو `@`؛ ثم الانحياز. هذا سطر الخلية العصبية بالكامل."),
                 ("14-15", "متجهات وحدة: نفس الاتجاه 1، متعامد 0، معاكس −1 — الضرب النقطي كمقياس تشابه."),
                 ("18-19", "`X @ w`: كل صف (ملاحظة) يُضرب نقطيًا في `w` دفعة واحدة؛ الشكل `(4,3)@(3,) → (4,)`. البث يضيف `b` للكل.")],
        run=run_printed(CODE),
        after_ar="- `X @ w + b` هو ما تفعله طبقة `Dense(1)` لدفعة كاملة.\n- الإشارة تحمل معنى: موجب = يشبه القالب.",
    ))
    common_mistake("`w * x` بدل `w @ x`: بلا خطأ تشغيل، لكن النتيجة متجه بدل عدد، ثم يفشل السطر التالي أو — أسوأ — يبثّ بصمت.")
    quiz("la.dot", [
        Q("$(1, 2, 3)\\cdot(4, 5, 6)$ يساوي…", ["32", "(4, 10, 18)", "21"], 0, "4+10+18.", kind="equation"),
        Q("ضرب نقطي = 0 يعني…", ["أحد المتجهين صفر", "المتجهان متعامدان (أو أحدهما صفر)", "المتجهان متساويان"], 1, "cos 90° = 0."),
        Q("`X.shape == (32, 5)`, `w.shape == (5,)`: `X @ w` شكله…", ["(32,)", "(5,)", "(32, 5)"], 0, "ضرب نقطي لكل صف.", kind="shape"),
    ])
    takeaway("w·x = Σ wⱼxⱼ رقم واحد يقيس التوافق. الخلية = ضرب نقطي + انحياز. @ لا *.")
    lesson_footer(LESSON, ["اضرب المتقابلات واجمع.", "إشارته وحجمه معنى: تشابه مع قالب الأوزان.", "X @ w يطبقه على الدفعة كلها."])
