import streamlit as st

from components.callouts import definition, research_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.calculus.jacobian_hessian",
    title_ar="تعمّق: الجاكوبي والهسي",
    title_en="Deep Dive: Jacobian & Hessian",
    module="foundations.calculus",
    order=5,
    prerequisites=["foundations.calculus.chain_rule", "foundations.linalg.matrix_multiplication"],
    objectives_ar=[
        "تعميم التدرج إلى الجاكوبي عندما يكون المخرج متجهًا.",
        "فهم الهسي كمصفوفة المشتقات الثانية ودورها في انحناء سطح الخسارة.",
        "معرفة لماذا لا تحسب الأطر الهسي كاملًا في الشبكات الكبيرة.",
    ],
    terms=["gradient"],
    difficulty="advanced",
    summary_ar="الجاكوبي: مشتقات مخرج متجهي؛ الهسي: مشتقات ثانية (انحناء). كلاهما مصفوفة.",
)

CODE = '''import numpy as np

# دالة بمخرج متجهي: طبقة خطية z = W x + b  (2 مخرج، 3 مدخل)
W = np.array([[1.0, -2.0, 0.5],
              [0.0,  3.0, 1.0]]); b = np.array([0.1, 0.2])
x = np.array([1.0, 2.0, -1.0])

def f(x): return W @ x + b

# الجاكوبي عدديًا: عمود j = ∂f/∂x_j
h = 1e-6
J = np.zeros((2, 3))
for j in range(3):
    e = np.zeros(3); e[j] = h
    J[:, j] = (f(x + e) - f(x - e)) / (2 * h)
print("Jacobian (numeric):"); print(J.round(4))
print("equals W?", np.allclose(J, W))

# الهسي لدالة عددية: L(w) = (w0-2)^2 + 3(w1+1)^2 + w0*w1
def L(w): return (w[0]-2)**2 + 3*(w[1]+1)**2 + w[0]*w[1]
w = np.array([0.0, 0.0])
H = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        ei = np.zeros(2); ej = np.zeros(2); ei[i] = h; ej[j] = h
        H[i, j] = (L(w+ei+ej) - L(w+ei-ej) - L(w-ei+ej) + L(w-ei-ej)) / (4*h*h)
print("Hessian (numeric):"); print(H.round(3))
print("eigenvalues (curvature):", np.linalg.eigvalsh(H).round(3))'''


def render() -> None:
    lesson_header(LESSON)
    research_note("هذا الدرس تعمّق اختياري. يكفي للمقرر أن تعرف الاسمين ومعناهما؛ لن تُطالَب بحسابهما يدويًا.")
    h2("الجاكوبي", "Jacobian")
    definition("عندما يكون **المخرج متجهًا** $\\mathbf{f}(\\mathbf{x}) \\in \\mathbb{R}^m$ والمدخل متجهًا $\\mathbf{x} \\in \\mathbb{R}^n$، فإن مصفوفة كل الجزئيات $J_{ij} = \\partial f_i / \\partial x_j$ بالشكل $(m, n)$ تسمى **الجاكوبي**. التدرج حالة خاصة عندما $m = 1$.")
    equation(r"J = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix}, \qquad \text{للطبقة الخطية } \mathbf{z} = W\mathbf{x} + \mathbf{b}:\ J = W",
             [("J_{ij}", "كيف يتغير المخرج $i$ عند تحريك المدخل $j$."), ("W", "جاكوبي الطبقة الخطية هو مصفوفة أوزانها نفسها.")],
             meaning_ar="قاعدة السلسلة بين الطبقات هي ضرب جاكوبيات: $\\frac{\\partial L}{\\partial \\mathbf{x}} = J^{\\mathsf T}\\frac{\\partial L}{\\partial \\mathbf{z}}$.",
             example_ar="طبقة من 3 مدخلات و2 مخرج: جاكوبي `(2, 3)`.",
             dl_link_ar="الانتشار الخلفي في طبقة خطية هو ضرب التدرج القادم في $W^{\\mathsf T}$ — لهذا ترى `W.T` في كود الانتشار الخلفي اليدوي.", title_ar="الجاكوبي")
    h2("الهسي", "Hessian")
    definition("لدالة **عددية** $L(\\mathbf{w})$، **الهسي** مصفوفة المشتقات الثانية $H_{ij} = \\partial^2 L / \\partial w_i \\partial w_j$ بالشكل $(n, n)$. يصف **انحناء** سطح الخسارة: قيمه الذاتية الموجبة = وعاء، السالبة = قمة، المختلطة = نقطة سرج.")
    equation(r"H = \nabla^2 L, \qquad L(\mathbf{w} + \Delta) \approx L(\mathbf{w}) + \nabla L^{\mathsf T}\Delta + \tfrac{1}{2}\Delta^{\mathsf T} H \Delta",
             [("H", "مصفوفة متماثلة $(n, n)$."), (r"\Delta", "خطوة صغيرة في فضاء المعلمات."), (r"\tfrac{1}{2}\Delta^{\mathsf T}H\Delta", "الحد التربيعي: الانحناء.")],
             meaning_ar="التدرج يعطي الميل؛ الهسي يعطي كيف يتغير الميل. طرق نيوتن تستخدمه لاختيار حجم الخطوة تلقائيًا.",
             example_ar="$L = (w_0-2)^2 + 3(w_1+1)^2 + w_0 w_1$: $H = \\begin{pmatrix}2 & 1\\\\1 & 6\\end{pmatrix}$.",
             dl_link_ar="بمليون معلمة الهسي له $10^{12}$ عنصر — مستحيل تخزينه. لهذا تعتمد الشبكات على طرق الرتبة الأولى (SGD, Adam) مع تقديرات تقريبية للانحناء (Adam يقرّبه عبر مربعات التدرج).", title_ar="الهسي")
    code_lab(CodeLab(
        key="calc_jh", title_ar="جاكوبي وهسي عدديًا", code=CODE,
        before=Before(goal_ar="حساب جاكوبي طبقة خطية عدديًا وإثبات أنه W، ثم هسي دالة خسارة صغيرة وقيمه الذاتية.", stage_ar="تفاضل متقدم.",
                      inputs_ar="مصفوفة W ودالة L بمتغيرين.", expected_ar="جاكوبي يساوي W، هسي [[2,1],[1,6]]، وقيم ذاتية موجبة (وعاء)."),
        explain=[("9-15", "عمود بعمود: حرّك مدخلًا واحدًا وقس تغير كل المخرجات."), ("16", "للطبقة الخطية الجاكوبي هو W بالضبط."),
                 ("19-26", "الهسي بفروق منتهية ثانية. متماثل لأن ترتيب الاشتقاق لا يهم."), ("27", "قيم ذاتية موجبة = وعاء = حد أدنى محلي حقيقي.")],
        run=run_printed(CODE),
        after_ar="- القيمتان الذاتيتان (≈1.76 و6.24) تقولان: الوعاء أشد انحناءً في اتجاه وأقل في آخر — سبب تعرج الانحدار التدريجي، وسبب فائدة تكييف معدل التعلم لكل معلمة (Adam).",
    ))
    quiz("calc.jh", [
        Q("جاكوبي دالة من 5 مدخلات إلى 3 مخرجات شكله…", ["(5, 3)", "(3, 5)", "(5, 5)"], 1, "(m, n) = (مخرجات، مدخلات).", kind="shape"),
        Q("هسي بقيم ذاتية مختلطة الإشارة يعني…", ["حد أدنى", "حد أقصى", "نقطة سرج"], 2, "انحناء لأعلى في اتجاه ولأسفل في آخر."),
        Q("لماذا لا تستخدم الشبكات الهسي كاملًا؟", ["لأنه غير دقيق", "لأن حجمه n² هائل", "لأنه دائمًا صفر"], 1, "10¹² عنصر لمليون معلمة."),
    ])
    takeaway("الجاكوبي (m, n) لمخرج متجهي — للطبقة الخطية هو W. الهسي (n, n) انحناء — كبير جدًا للشبكات، فتقرّبه طرق مثل Adam.")
    lesson_footer(LESSON, ["الجاكوبي يعمم التدرج.", "الهسي يصف الانحناء ونقاط السرج.", "طرق الرتبة الأولى هي العملية في التعلم العميق."])
