import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.linalg.scalars_vectors",
    title_ar="الأعداد والمتجهات وعملياتها",
    title_en="Scalars, Vectors & Vector Operations",
    module="foundations.linalg",
    order=1,
    prerequisites=["foundations.math.notation", "foundations.python.numpy_ndarray"],
    objectives_ar=[
        "تعريف العدد (رتبة 0) والمتجه (رتبة 1) ومكوناته.",
        "الجمع والطرح والضرب في عدد هندسيًا وعدديًا.",
        "حساب طول المتجه (المعيار) وفهم ملاحظة = متجه، تدرج = متجه.",
    ],
    terms=["tensor", "rank", "shape", "gradient"],
    labs=["labs.vector_matrix"],
    difficulty="beginner",
    summary_ar="العدد رقم واحد، المتجه قائمة مرتبة من الأرقام؛ كل ملاحظة متجه وكل تدرج متجه.",
)

CODE = '''import numpy as np

x = np.array([3.0, 1.0])       # متجه بمكونين (ملاحظة بخاصيتين)
y = np.array([1.0, 2.0])
c = 2.0                        # عدد (scalar)

print(x + y, x - y)            # جمع/طرح: مكوّنة بمكوّنة
print(c * x)                   # ضرب في عدد: تمديد
print(np.linalg.norm(x))       # الطول (المعيار الإقليدي): sqrt(3² + 1²)
print(np.sqrt((x ** 2).sum())) # نفس الطول يدويًا
print(x / np.linalg.norm(x))   # متجه وحدة (طوله 1) في نفس الاتجاه
print(x.shape, c, np.array(c).shape)   # (2,) مقابل ()'''


def _plot(x, y):
    fig = go.Figure()
    for v, name, color in [(x, "x", "#2F6FB5"), (y, "y", "#7C5CBF"), (x + y, "x + y", "#1F7A78")]:
        fig.add_trace(go.Scatter(x=[0, v[0]], y=[0, v[1]], mode="lines+markers", name=name, line=dict(color=color, width=3)))
    fig.add_trace(go.Scatter(x=[x[0], x[0] + y[0]], y=[x[1], x[1] + y[1]], mode="lines", showlegend=False, line=dict(color="#7C5CBF", dash="dot")))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), xaxis=dict(range=[-1, 6], zeroline=True), yaxis=dict(range=[-1, 5], scaleanchor="x"),
                      plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="vec_plot")


def render() -> None:
    lesson_header(LESSON)
    h2("العدد والمتجه", "Scalar & vector")
    definition("**العدد** `Scalar` رقم واحد (رتبة 0، شكله `()`)، مثل معدل التعلم أو الخسارة. **المتجه** `Vector` قائمة مرتبة من $n$ أرقام (رتبة 1، شكله `(n,)`)؛ كل رقم **مكوّنة** `Component`.")
    intuition("ملاحظة بخاصيتين (ساعات المذاكرة، ساعات النوم) هي نقطة في مستوى — أو سهم من الأصل إلى تلك النقطة. بثلاث خصائص: نقطة في الفضاء. بـ 784 خاصية (صورة 28×28): نقطة في فضاء لا نتخيله لكن الحساب نفسه.")
    table(["الكائن", "الرتبة", "الشكل", "أمثلة في التعلم العميق"],
          [("عدد", "0", "()", "learning_rate، loss، b في خلية واحدة"), ("متجه", "1", "(n,)", "ملاحظة x، متجه الهدف y، الانحياز b لطبقة، التدرج ∇L")],
          ["rtl", "num", "code", "rtl"])
    h2("العمليات", "Operations")
    equation(r"\mathbf{x} + \mathbf{y} = (x_1 + y_1,\ \dots,\ x_n + y_n), \qquad c\,\mathbf{x} = (c x_1,\ \dots,\ c x_n)",
             [(r"\mathbf{x}, \mathbf{y}", "متجهان بنفس الطول (شرط الجمع)."), ("c", "عدد يمدّد أو يقلّص المتجه (ويعكسه إن كان سالبًا).")],
             meaning_ar="الجمع مكوّنة بمكوّنة (هندسيًا: ذيل الثاني على رأس الأول)؛ الضرب في عدد يغيّر الطول لا الاتجاه.",
             example_ar="$(3, 1) + (1, 2) = (4, 3)$؛ $2 \\cdot (3, 1) = (6, 2)$.",
             dl_link_ar="تحديث المعلمات $\\mathbf{w} \\leftarrow \\mathbf{w} - \\eta \\nabla L$ هو: متجه ناقص (عدد × متجه).", title_ar="جمع المتجهات والضرب في عدد")
    h3("جرّب", "Explore")
    c1, c2 = st.columns(2)
    with c1:
        x1 = st.slider("x₁", -2.0, 4.0, 3.0, 0.5, key="vec_x1"); x2 = st.slider("x₂", -2.0, 4.0, 1.0, 0.5, key="vec_x2")
    with c2:
        y1 = st.slider("y₁", -2.0, 4.0, 1.0, 0.5, key="vec_y1"); y2 = st.slider("y₂", -2.0, 4.0, 2.0, 0.5, key="vec_y2")
    x = np.array([x1, x2]); y = np.array([y1, y2])
    _plot(x, y)
    st.code(f"x + y = {x + y}   |x| = {np.linalg.norm(x):.3f}   |y| = {np.linalg.norm(y):.3f}", language="text")
    equation(r"\|\mathbf{x}\| = \sqrt{\sum_{i=1}^{n} x_i^2}", [(r"\|\mathbf{x}\|", "طول المتجه (المعيار الإقليدي).")],
             meaning_ar="فيثاغورس في $n$ بُعد.", example_ar="$\\|(3, 4)\\| = \\sqrt{9 + 16} = 5$.",
             dl_link_ar="معيار التدرج `gradient norm` يخبرك إن كانت التدرجات تتلاشى (قريبة من 0) أو تنفجر (ضخمة). قصّ التدرج يحدّ من هذا المعيار.", title_ar="الطول (المعيار)")
    code_lab(CodeLab(
        key="la_vectors", title_ar="متجهات في NumPy", code=CODE,
        before=Before(goal_ar="تنفيذ العمليات الأساسية وحساب الطول.", stage_ar="جبر خطي أساسي.", inputs_ar="متجهان بطول 2 وعدد.", expected_ar="نتائج مكوّنة بمكوّنة، الطول √10 ≈ 3.162، متجه وحدة."),
        explain=[("3-5", "المتجه `(2,)` والعدد بشكل `()`."), ("7-8", "العمليات عنصرية تلقائيًا."), ("9-11", "`norm` = الجذر التربيعي لمجموع المربعات؛ القسمة عليه تعطي متجهًا طوله 1 (تطبيع)."), ("12", "الفرق بين `(2,)` و`()`: رتبة 1 مقابل رتبة 0.")],
        run=run_printed(CODE),
        after_ar="- `x / norm(x)` تطبيع الاتجاه: يظهر في تطبيع التضمينات وفي بعض دوال التنشيط.\n- شكل العدد `()` — صف فارغ، رتبة صفر.",
    ))
    common_mistake("جمع متجهين بطولين مختلفين `(3,) + (2,)` خطأ؛ الجمع يتطلب نفس الشكل (أو بثًا صالحًا).")
    quiz("la.vectors", [
        Q("رتبة العدد 3.7 هي…", ["0", "1", "()"], 0, "عدد = رتبة 0."),
        Q("$2\\cdot(1, -3)$ يساوي…", ["(2, -6)", "(3, -1)", "(2, 6)"], 0, "ضرب كل مكوّنة.", kind="equation"),
        Q("$\\|(6, 8)\\|$ يساوي…", ["14", "10", "48"], 1, "√(36+64).", kind="equation"),
    ])
    takeaway("العدد رقم، المتجه قائمة مرتبة. الجمع مكوّنة بمكوّنة، الضرب في عدد يمدّد، والمعيار طول. تحديث الأوزان جبر متجهات.")
    lesson_footer(LESSON, ["رتبة 0 و1.", "x + y و c·x عنصرية.", "‖x‖ = √Σx²؛ معيار التدرج أداة تشخيص."])
