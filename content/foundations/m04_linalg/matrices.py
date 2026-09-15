import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.linalg.matrices",
    title_ar="المصفوفات: الصفوف والأعمدة والشكل والمنقولة",
    title_en="Matrices: Rows, Columns, Shape & Transpose",
    module="foundations.linalg",
    order=3,
    prerequisites=["foundations.linalg.dot_product"],
    objectives_ar=[
        "قراءة مصفوفة $A \\in \\mathbb{R}^{m\\times n}$ وعنصرها $a_{ij}$.",
        "الجمع والضرب في عدد والمنقولة، والتمييز بين المنقولة وإعادة التشكيل.",
        "معرفة أن X (ملاحظات × خصائص) وW (مدخلات × مخرجات) مصفوفات ذات أدوار مختلفة.",
    ],
    terms=["shape", "axis"],
    labs=["labs.vector_matrix"],
    difficulty="beginner",
    summary_ar="المصفوفة شبكة m×n؛ a_ij صف i عمود j؛ المنقولة تبدّل الصفوف بالأعمدة (ليست reshape).",
)

CODE = '''import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])            # 2×3: m=2 صفوف، n=3 أعمدة
print(A.shape, A[0, 2], A[1, 0])    # a_13 = 3, a_21 = 4 (بفهرسة تبدأ من 0)

B = np.ones((2, 3))
print(A + B)                        # جمع: نفس الشكل، عنصر بعنصر
print(0.5 * A)                      # ضرب في عدد

print(A.T, A.T.shape)               # المنقولة: 3×2، a_ij ↔ a_ji
print(A.reshape(3, 2))              # إعادة تشكيل: نفس الترتيب في الذاكرة — مختلف تمامًا!

I = np.eye(3)                       # مصفوفة الوحدة 3×3
print(I)

X = np.array([[4200, 34], [6100, 45], [2900, 29]], dtype=float)   # 3 ملاحظات × 2 خاصية
print("features as rows:", X.T.shape)   # المنقولة: خاصية في كل صف'''


def render() -> None:
    lesson_header(LESSON)
    h2("ما هي المصفوفة؟", "What is a matrix?")
    definition("**المصفوفة** `Matrix` شبكة مستطيلة من الأرقام بـ $m$ صفوف و$n$ أعمدة (رتبة 2، شكل `(m, n)`). العنصر $a_{ij}$ في الصف $i$ والعمود $j$ — **الصف أولًا**.")
    intuition("جدول البيانات مصفوفة: كل صف ملاحظة وكل عمود خاصية. مصفوفة الأوزان جدول آخر: كل صف مدخل وكل عمود وحدة مخرجة.")
    table(["المصفوفة", "الشكل", "الصفوف تعني", "الأعمدة تعني"],
          [("X (البيانات)", "(n, d)", "ملاحظات", "خصائص"), ("W (الأوزان)", "(d, k)", "مدخلات الطبقة", "وحدات الطبقة"),
           ("مصفوفة الالتباس", "(k, k)", "الفئة الحقيقية", "الفئة المتنبأة"), ("صورة رمادية", "(H, W)", "صفوف بكسل", "أعمدة بكسل")],
          ["ltr", "code", "rtl", "rtl"])
    h2("العمليات الأساسية", "Basic operations")
    equation(r"(A + B)_{ij} = a_{ij} + b_{ij}, \qquad (cA)_{ij} = c\,a_{ij}, \qquad (A^{\mathsf T})_{ij} = a_{ji}",
             [("A + B", "الجمع يتطلب نفس الشكل."), ("cA", "الضرب في عدد يضرب كل عنصر."), (r"A^{\mathsf T}", "المنقولة: تبديل الصفوف بالأعمدة؛ الشكل `(m, n)` يصبح `(n, m)`.")],
             meaning_ar="كلها عنصرية ما عدا المنقولة التي تعيد ترتيب المواضع.",
             example_ar="$A = \\begin{pmatrix}1&2&3\\\\4&5&6\\end{pmatrix}$ ⟹ $A^{\\mathsf T} = \\begin{pmatrix}1&4\\\\2&5\\\\3&6\\end{pmatrix}$.",
             dl_link_ar="في `PyTorch` تُخزَّن الأوزان بالشكل `(out, in)` وتُحسب الطبقة كـ $xW^{\\mathsf T}$، بينما في `Keras` `(in, out)` وتُحسب $xW$. نفس الرياضيات، اتفاق مختلف.", title_ar="جمع، ضرب في عدد، منقولة")
    code_lab(CodeLab(
        key="la_matrices", title_ar="مصفوفات: شكل، عناصر، منقولة، وحدة", code=CODE,
        before=Before(goal_ar="التعامل مع المصفوفة كشبكة والتمييز بين المنقولة وإعادة التشكيل.", stage_ar="جبر خطي.", inputs_ar="مصفوفة 2×3 وأخرى 3×2.",
                      expected_ar="الشكل والعناصر، جمع وضرب عنصريان، منقولة 3×2 تختلف عن `reshape(3, 2)`، ومصفوفة وحدة."),
        explain=[("3-5", "الفهرسة `[صف, عمود]`؛ `A[0, 2]` هو $a_{13}$ رياضيًا (الفرق بين العدّ من 0 ومن 1)."),
                 ("7-9", "جمع وضرب في عدد: عنصريان."),
                 ("11-12", "المنقولة تبدّل المحورين فيصبح العمود صفًا. `reshape` يقرأ العناصر بالترتيب ويصبّها في شكل جديد — النتيجتان مختلفتان! قارنهما في المخرجات."),
                 ("14-15", "مصفوفة الوحدة: آحاد على القطر؛ ضرب أي مصفوفة فيها يتركها كما هي (كـ «1» للمصفوفات)."),
                 ("17-18", "`X.T` يجعل كل خاصية صفًا — مفيد أحيانًا للحساب، لكن الاتفاق في التعلم العميق: الملاحظات صفوف.")],
        run=run_printed(CODE),
        after_ar="- `A.T` أول صف `[1, 4]` بينما `A.reshape(3, 2)` أول صف `[1, 2]`: المنقولة تعيد الترتيب، وإعادة التشكيل تحافظ عليه.",
    ))
    common_mistake("استخدام `reshape` لتحويل `(n, d)` إلى `(d, n)` ظنًا أنه المنقولة. الشكل صحيح، البيانات مبعثرة، ولا رسالة خطأ.")
    quiz("la.matrices", [
        Q("في مصفوفة `(5, 3)`، $a_{31}$ هو…", ["الصف 3 العمود 1", "الصف 1 العمود 3", "العنصر 31"], 0, "الصف أولًا."),
        Q("منقولة `(4, 2)` شكلها…", ["(4, 2)", "(2, 4)", "(8,)"], 1, "تبديل المحورين.", kind="shape"),
        Q("`A.T` مقابل `A.reshape(n, m)`…", ["متطابقان", "نفس الشكل، ترتيب مختلف للعناصر", "شكلان مختلفان"], 1, "المنقولة ليست إعادة تشكيل."),
    ])
    takeaway("المصفوفة شبكة (m, n)؛ a_ij صف ثم عمود. الجمع والضرب في عدد عنصريان؛ المنقولة تبدّل المحاور ولا تساوي reshape.")
    lesson_footer(LESSON, ["X (ملاحظات × خصائص)، W (مدخلات × وحدات).", "Aᵀ يبدّل الصفوف والأعمدة.", "reshape ≠ transpose."])
