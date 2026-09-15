import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.math.functions",
    title_ar="الدالة: مدخل ← مخرج، المجال والمدى، خطي ولاخطي",
    title_en="Functions: Input → Output, Domain/Range, Linear vs Nonlinear",
    module="foundations.math",
    order=2,
    prerequisites=["foundations.math.notation"],
    objectives_ar=[
        "تعريف الدالة كقاعدة تربط كل مدخل بمخرج واحد.",
        "قراءة المجال والمدى، وربطهما بمديات دوال التنشيط لاحقًا.",
        "التمييز بيانيًا بين الخطي واللاخطي، وفهم لماذا تحتاج الشبكة لاخطية.",
    ],
    terms=["model"],
    difficulty="beginner",
    summary_ar="الدالة آلة مدخل ← مخرج؛ النموذج دالة؛ الخطي خط مستقيم واللاخطي كل ما عداه.",
)


def _plot(kind: str, a: float, b: float) -> None:
    x = np.linspace(-4, 4, 200)
    if kind == "خطي: f(x) = a·x + b":
        y = a * x + b
    elif kind == "تربيعي: f(x) = a·x² + b":
        y = a * x ** 2 + b
    elif kind == "سيجمويد: f(x) = 1 / (1 + e^(−a·x)) + b":
        y = 1 / (1 + np.exp(-a * x)) + b
    else:
        y = np.maximum(0, a * x) + b
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#1F7A78", width=3)))
    fig.add_hline(y=0, line=dict(color="#B9B2A6", width=1)); fig.add_vline(x=0, line=dict(color="#B9B2A6", width=1))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="x (input)", yaxis_title="f(x) (output)",
                      plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", title=dict(text=kind, x=0.5))
    st.plotly_chart(fig, width="stretch", key="fn_plot")
    st.code(f"x = 2  →  f(2) = {float(np.interp(2, x, y)):.3f}\nx = -2 →  f(-2) = {float(np.interp(-2, x, y)):.3f}\nالمدى المرسوم: [{y.min():.2f}, {y.max():.2f}]", language="text")


def render() -> None:
    lesson_header(LESSON)
    h2("ما هي الدالة؟", "What is a function?")
    definition("**الدالة** $f$ قاعدة تعطي لكل مدخل $x$ مخرجًا واحدًا $f(x)$. **المجال** `Domain` مجموعة المدخلات المسموحة، **المدى** `Range` مجموعة المخرجات الممكنة.")
    intuition("آلة بيع: تُدخل رقم الزر (المدخل) فتحصل على منتج واحد محدد (المخرج). نفس الزر لا يعطي منتجين مختلفين — هذا شرط الدالة.")
    why("النموذج بأكمله دالة: $\\hat{y} = f_\\theta(x)$. كل طبقة دالة، ودالة التنشيط دالة، والخسارة دالة في المعلمات. تعلّم قراءة الدوال = تعلّم قراءة الشبكة.")
    h2("خطي أم لاخطي؟", "Linear or nonlinear?")
    equation(r"f(x) = a\,x + b", [("a", "الميل: كم يتغير المخرج عند زيادة المدخل بوحدة."), ("b", "التقاطع: المخرج عند $x = 0$.")],
             meaning_ar="خط مستقيم. مضاعفة المدخل تضاعف الأثر دائمًا بنفس النسبة.",
             example_ar="$a=2, b=1$: $f(3) = 7$، $f(6) = 13$. الفرق ثابت لكل وحدة.",
             dl_link_ar="الطبقة الكثيفة بلا تنشيط دالة خطية. تراكب دوال خطية يبقى خطيًا — لذلك نحتاج التنشيط اللاخطي.", title_ar="الدالة الخطية")
    h3("جرّب بنفسك", "Explore")
    kind = st.selectbox("الدالة", ["خطي: f(x) = a·x + b", "تربيعي: f(x) = a·x² + b", "سيجمويد: f(x) = 1 / (1 + e^(−a·x)) + b", "ReLU: f(x) = max(0, a·x) + b"], key="fn_kind")
    c1, c2 = st.columns(2)
    with c1:
        a = st.slider("a", -3.0, 3.0, 1.0, 0.1, key="fn_a")
    with c2:
        b = st.slider("b", -3.0, 3.0, 0.0, 0.1, key="fn_b")
    _plot(kind, a, b)
    table(
        ["الدالة", "المجال", "المدى", "خطية؟", "أين تظهر"],
        [("a·x + b", "كل الأعداد", "كل الأعداد", "نعم", "الطبقة الكثيفة قبل التنشيط"),
         ("x²", "كل الأعداد", "[0, ∞)", "لا", "مربع الخطأ في MSE"),
         ("1/(1+e^(−x))", "كل الأعداد", "(0, 1)", "لا", "Sigmoid: احتمال"),
         ("max(0, x)", "كل الأعداد", "[0, ∞)", "لا (مكسورة)", "ReLU"),
         ("log(x)", "(0, ∞)", "كل الأعداد", "لا", "الإنتروبيا المتقاطعة")],
        ["code", "rtl", "code", "rtl", "rtl"],
    )
    common_mistake("«المدى» يخبرك لماذا Sigmoid مناسبة لمخرج احتمالي (0..1) وغير مناسبة لانحدار قيمته 5000. اختيار تنشيط طبقة الإخراج هو اختيار مدى.")
    quiz("math.functions", [
        Q("مدى $f(x) = x^2$ هو…", ["كل الأعداد", "[0, ∞)", "(0, 1)"], 1, "المربع لا يكون سالبًا."),
        Q("لماذا تراكب طبقتين خطيتين لا يزيد قدرة النموذج؟", ["لأن الحساب بطيء", "لأن تركيب خطيين خطي", "لأن الأوزان تتساوى"], 1, "خطي ∘ خطي = خطي."),
        Q("مخرج يجب أن يكون احتمالًا. أي مدى تريده؟", ["كل الأعداد", "(0, 1)", "[0, ∞)"], 1, "Sigmoid/Softmax."),
    ])
    takeaway("الدالة مدخل ← مخرج واحد. المدى يحدد أي تنشيط يناسب أي مهمة. اللاخطية ضرورة لا رفاهية.")
    lesson_footer(LESSON, ["دالة = قاعدة واحدة لكل مدخل.", "المجال مدخلات، المدى مخرجات.", "خطي ∘ خطي = خطي ← نحتاج تنشيطًا لاخطيًا."])
