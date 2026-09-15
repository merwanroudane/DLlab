import numpy as np
import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.diagram import diagram, svg_box, svg_text
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation, worked_steps
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.linalg.matrix_multiplication",
    title_ar="ضرب المصفوفات، ضرب هادامارد، ومصفوفة الوحدة",
    title_en="Matrix Multiplication, Hadamard Product & Identity",
    module="foundations.linalg",
    order=4,
    prerequisites=["foundations.linalg.matrices"],
    objectives_ar=[
        "حساب $C = AB$ كضرب نقطي لكل (صف من A، عمود من B) وتطبيق شرط التوافق `(m, n)@(n, k) → (m, k)`.",
        "التمييز بين `@` (ضرب مصفوفات) و`*` (هادامارد، عنصري).",
        "حساب شكل مخرج طبقة كثيفة وعدد معلماتها من شكل W.",
    ],
    terms=["weight", "bias", "batch_dimension"],
    labs=["labs.vector_matrix"],
    difficulty="intermediate",
    summary_ar="AB: ضرب نقطي لكل صف بكل عمود؛ الشرط (m,n)@(n,k). الطبقة الكثيفة = XW + b.",
)

CODE = '''import numpy as np

X = np.array([[1., 2.],
              [3., 4.],
              [5., 6.]])                 # (3, 2): 3 ملاحظات × 2 خاصية
W = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6]])          # (2, 3): 2 مدخل × 3 وحدات
b = np.array([0.01, 0.02, 0.03])          # (3,): انحياز لكل وحدة

Z = X @ W + b                             # (3,2)@(2,3) → (3,3)
print(Z.shape); print(Z.round(3))

# العنصر Z[0, 0] يدويًا: الصف 0 من X · العمود 0 من W + b[0]
print(X[0] @ W[:, 0] + b[0])

# هادامارد (عنصري) يتطلب نفس الشكل
print(X * X)                              # (3,2)*(3,2)
try:
    print(X * W)                          # (3,2)*(2,3) → خطأ بث
except ValueError as e:
    print("ValueError:", e)

print(np.allclose(X @ np.eye(2), X))      # مصفوفة الوحدة لا تغيّر شيئًا
print((X @ W).shape, "params =", W.size + b.size)   # عدد معلمات الطبقة = 2*3 + 3'''


def _shape_svg() -> str:
    s = '<svg viewBox="0 0 560 150" width="100%" style="max-width:560px">'
    s += svg_box(20, 40, 110, 60, "X  (3, 2)", "#E6F1FB", stroke="#2F6FB5", font=14, bold=True)
    s += svg_text(150, 75, "@", size=22, bold=True)
    s += svg_box(175, 40, 110, 60, "W  (2, 3)", "#EFE9FA", stroke="#7C5CBF", font=14, bold=True)
    s += svg_text(305, 75, "=", size=22, bold=True)
    s += svg_box(330, 40, 110, 60, "Z  (3, 3)", "#DDF5EA", stroke="#2E8B57", font=14, bold=True)
    s += svg_text(153, 125, "inner dims must match: 2 = 2", size=12, color="#C8473A")
    s += svg_text(385, 125, "outer dims survive: 3 × 3", size=12, color="#2E8B57")
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("التعريف", "Definition")
    equation(r"C = AB, \qquad c_{ij} = \sum_{l=1}^{n} a_{il}\, b_{lj}, \qquad A\in\mathbb{R}^{m\times n},\ B\in\mathbb{R}^{n\times k},\ C\in\mathbb{R}^{m\times k}",
             [("c_{ij}", "العنصر في الصف $i$ والعمود $j$ من الناتج."), ("a_{il}", "الصف $i$ من $A$."), ("b_{lj}", "العمود $j$ من $B$."), ("n", "البُعد الداخلي: يجب أن يتطابق (أعمدة $A$ = صفوف $B$).")],
             meaning_ar="كل عنصر في الناتج هو **ضرب نقطي** لصف من الأولى في عمود من الثانية.",
             example_ar="$\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}\\begin{pmatrix}5&6\\\\7&8\\end{pmatrix} = \\begin{pmatrix}1·5+2·7 & 1·6+2·8\\\\3·5+4·7 & 3·6+4·8\\end{pmatrix} = \\begin{pmatrix}19&22\\\\43&50\\end{pmatrix}$.",
             dl_link_ar="الطبقة الكثيفة لدفعة: $Z = XW + b$ حيث $X$ `(n, d)` و$W$ `(d, k)` فيكون $Z$ `(n, k)`: كل صف ملاحظة، كل عمود وحدة.", title_ar="ضرب المصفوفات")
    diagram("قاعدة الأشكال", _shape_svg(), what_ar="ثلاث مصفوفات بأشكالها. البُعدان الداخليان (2 و2) يجب أن يتساويا ويختفيان؛ الخارجيان (3 و3) يبقيان.",
            how_ar="اكتب الشكلين متجاورين (3,2)(2,3): إن تطابق الرقمان المتوسطان فالضرب ممكن والنتيجة (الأول, الأخير).",
            takeaway_ar="(n, d) @ (d, k) → (n, k). هذه القاعدة تحسب شكل كل طبقة في أي شبكة.", title_en="Shape rule")
    intuition("كل وحدة في الطبقة لها عمود أوزان. `X @ W` يجري الضرب النقطي لكل ملاحظة مع كل وحدة دفعة واحدة — $n \\times k$ ضربًا نقطيًا في عملية واحدة.")
    worked_steps([("الصف 0 من X والعمود 0 من W", r"(1, 2)\cdot(0.1, 0.4) = 0.1 + 0.8 = 0.9"),
                  ("أضف الانحياز b₀", r"z_{00} = 0.9 + 0.01 = 0.91"),
                  ("كرر لكل (صف، عمود)", r"Z \in \mathbb{R}^{3\times 3}")], title_ar="حساب عنصر واحد")
    h2("`@` مقابل `*`", "Matrix product vs Hadamard")
    compare_table(["", "ضرب المصفوفات", "هادامارد (عنصري)"],
                  [("الرمز", "@ أو np.matmul / np.dot", "*"), ("الشرط", "(m, n)@(n, k)", "نفس الشكل (أو بث)"), ("الناتج", "(m, k)", "نفس الشكل"),
                   ("المعنى", "تركيب تحويل خطي / طبقة", "ضرب مقابل بمقابل: أقنعة، بوابات LSTM، Dropout"), ("رياضيًا", "AB", "A ⊙ B")],
                  ["rtl", "code", "code"])
    why("في LSTM: $f_t \\odot c_{t-1}$ هادامارد (البوابة تضرب كل مكوّنة في احتمالها)، بينما $W_f x_t$ ضرب مصفوفات. الخلط بينهما يعطي خطأ شكل أو معنى.")
    code_lab(CodeLab(
        key="la_matmul", title_ar="طبقة كثيفة = X @ W + b", code=CODE,
        before=Before(goal_ar="حساب طبقة كثيفة لدفعة، التحقق من عنصر يدويًا، ورؤية الفرق بين @ و*.", stage_ar="جبر خطي ← الطبقة.",
                      inputs_ar="`X (3, 2)`، `W (2, 3)`، `b (3,)`.", expected_ar="`Z (3, 3)`، عنصر `0.91`، هادامارد ينجح لنفس الشكل ويفشل لـ (3,2)*(2,3)، ومعلمات = 9.",
                      math_ar="$Z = XW + b$، عدد المعلمات $= d\\,k + k$."),
        explain=[("3-8", "الملاحظات صفوف X، الوحدات أعمدة W، انحياز لكل وحدة."),
                 ("10-11", "الضرب ثم بث الانحياز عبر الصفوف. `(3,2)@(2,3)` → `(3,3)`."),
                 ("14", "التحقق اليدوي: صف في عمود + انحياز."),
                 ("17-21", "هادامارد يحتاج نفس الشكل؛ `(3,2)*(2,3)` يفشل بالبث — وهو الخطأ الذي يظهر عند استخدام `*` بدل `@`."),
                 ("23-24", "مصفوفة الوحدة محايدة؛ وعدد معلمات الطبقة = عناصر W + عناصر b = 2×3 + 3 = 9 — نفس الرقم الذي يطبعه `model.summary()`.")],
        run=run_printed(CODE),
        after_ar="- الشكل `(3, 3)`: 3 ملاحظات × 3 وحدات.\n- `Dense(3)` على مدخل بطول 2 لها 9 معلمات: `2*3 + 3`. هكذا تتحقق من `Param #` لاحقًا.",
    ))
    h3("حاسبة الطبقة", "Layer calculator")
    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.number_input("n ملاحظات في الدفعة", 1, 512, 32, key="mm_n")
    with c2:
        d = st.number_input("d مدخلات", 1, 4096, 4, key="mm_d")
    with c3:
        k = st.number_input("k وحدات", 1, 4096, 64, key="mm_k")
    st.code(f"X: ({n}, {d})  @  W: ({d}, {k})  +  b: ({k},)   →   Z: ({n}, {k})\nparams = d*k + k = {d}*{k} + {k} = {d * k + k}", language="text")
    debugging_note("`ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0` أو في Keras `expected axis -1 of input shape to have value 4 but received ... 3`: البُعد الداخلي لا يتطابق. غالبًا `input_shape` خاطئ أو نسيت ترميز/حذف عمود.")
    common_mistake("`W @ X` بدل `X @ W`: الضرب غير تبادلي. `(2,3)@(3,2)` ينجح ويعطي `(2,2)` بمعنى خاطئ تمامًا.")
    quiz("la.matmul", [
        Q("`(64, 10) @ (10, 5)` يعطي…", ["(64, 5)", "(10, 10)", "خطأ"], 0, "الداخليان يختفيان.", kind="shape"),
        Q("`(64, 10) @ (5, 10)`…", ["(64, 10)", "(64, 5)", "خطأ: 10 ≠ 5"], 2, "لا تطابق.", kind="shape"),
        Q("طبقة `Dense(16)` على مدخل بطول 8: كم معلمة؟", ["128", "144", "24"], 1, "8×16 + 16.", kind="shape"),
        Q("أي عملية تستخدمها بوابة النسيان في LSTM على حالة الخلية؟", ["@", "* (هادامارد)", "+"], 1, "عنصرية."),
    ])
    takeaway("AB = ضرب نقطي لكل صف بكل عمود؛ (m,n)@(n,k)→(m,k)؛ غير تبادلي. * عنصري. الطبقة = XW + b بمعلمات dk + k.")
    lesson_footer(LESSON, ["قاعدة الأشكال تحسب كل طبقة.", "@ ≠ *.", "Param # = d·k + k."])
