import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import definition, intuition, takeaway, why
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation, worked_steps
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto

LESSON = Lesson(
    id="foundations.calculus.change_slope",
    title_ar="التغير ومعدل التغير والميل: القاطع والمماس",
    title_en="Change, Rate of Change & Slope: Secant and Tangent",
    module="foundations.calculus",
    order=1,
    prerequisites=["foundations.math.functions"],
    objectives_ar=[
        "حساب معدل التغير بين نقطتين كميل القاطع.",
        "رؤية كيف يصبح القاطع مماسًا عندما تقترب النقطتان.",
        "قراءة الميل كإجابة عن «إن زدت المدخل قليلًا، كم يتغير المخرج؟»",
    ],
    terms=["gradient"],
    labs=["labs.derivative_lab"],
    difficulty="beginner",
    summary_ar="الميل = التغير في المخرج / التغير في المدخل؛ القاطع بين نقطتين، والمماس عندما تلتصقان.",
)


def _f(x):
    return 0.5 * x ** 2


def render() -> None:
    lesson_header(LESSON)
    h2("لماذا نبدأ من الميل؟", "Why slope?")
    why("السؤال الوحيد الذي يطرحه التدريب مليارات المرات: «إن غيّرت هذا الوزن قليلًا، هل تزيد الخسارة أم تنقص، وبكم؟». هذا السؤال هو الميل.")
    definition("**معدل التغير** بين نقطتين $x_1$ و$x_2$ هو نسبة التغير في المخرج إلى التغير في المدخل. هندسيًا هو **ميل القاطع** `Secant` الذي يمر بالنقطتين.")
    equation(r"\text{slope} = \frac{\Delta y}{\Delta x} = \frac{f(x_2) - f(x_1)}{x_2 - x_1}",
             [(r"\Delta y", "التغير في المخرج (دلتا = فرق)."), (r"\Delta x", "التغير في المدخل."), ("f(x_2) - f(x_1)", "الفرق بين قيمتي الدالة.")],
             meaning_ar="كم وحدة يتغير المخرج لكل وحدة تغير في المدخل.",
             example_ar="$f(x) = 0.5x^2$: بين $x=1$ و$x=3$: $\\frac{4.5 - 0.5}{2} = 2$.",
             dl_link_ar="إن كانت $f$ هي الخسارة و$x$ وزنًا، فالميل يخبرنا اتجاه تعديل الوزن: ميل موجب ← قلّل الوزن.", title_ar="ميل القاطع")
    h3("من القاطع إلى المماس", "Secant → tangent")
    x0 = st.slider("النقطة x₀", -3.0, 3.0, 1.0, 0.25, key="slope_x0")
    h = st.select_slider("المسافة h بين النقطتين", options=[2.0, 1.0, 0.5, 0.25, 0.1, 0.01], value=1.0, key="slope_h")
    xs = np.linspace(-3.5, 3.5, 200)
    sec = (_f(x0 + h) - _f(x0)) / h
    tan = x0  # derivative of 0.5x^2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=_f(xs), mode="lines", name="f(x) = 0.5x²", line=dict(color="#2F6FB5", width=3)))
    fig.add_trace(go.Scatter(x=[x0, x0 + h], y=[_f(x0), _f(x0 + h)], mode="markers", name="النقطتان", marker=dict(size=10, color="#C8473A")))
    fig.add_trace(go.Scatter(x=xs, y=_f(x0) + sec * (xs - x0), mode="lines", name=f"القاطع (ميل {sec:.3f})", line=dict(color="#C8473A", dash="dash")))
    fig.add_trace(go.Scatter(x=xs, y=_f(x0) + tan * (xs - x0), mode="lines", name=f"المماس (ميل {tan:.3f})", line=dict(color="#1F7A78")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[-1, 7]), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="slope_fig")
    st.code(f"secant slope = (f({x0 + h:.2f}) - f({x0:.2f})) / {h} = {sec:.4f}\ntangent slope at x0 = {tan:.4f}\ndifference = {abs(sec - tan):.4f}  ← يتناقص كلما صغرت h", language="text")
    intuition("صغّر `h` وراقب القاطع يلتصق بالمماس. **المماس** `Tangent` هو الخط الذي يلمس المنحنى في نقطة واحدة ويصف اتجاهه هناك. ميله هو ما سنسميه في الدرس التالي **المشتقة**.")
    worked_steps([("اختر $x_0 = 1$ و$h = 0.5$", r"f(1) = 0.5,\ f(1.5) = 1.125"), ("ميل القاطع", r"\frac{1.125 - 0.5}{0.5} = 1.25"),
                  ("بـ $h = 0.1$", r"\frac{f(1.1) - f(1)}{0.1} = \frac{0.605 - 0.5}{0.1} = 1.05"), ("النهاية عند $h \\to 0$", r"1.00 = \text{ميل المماس}")])
    st.button("افتح معمل المشتقة", icon=":material/science:", on_click=goto, args=("labs.derivative_lab",), key="slope_lab")
    quiz("calc.slope", [
        Q("$f(x) = x^2$، بين $x = 2$ و$x = 4$ ميل القاطع…", ["6", "12", "2"], 0, "(16 − 4)/2.", kind="equation"),
        Q("كلما صغرت $h$…", ["يبتعد القاطع عن المماس", "يقترب القاطع من المماس", "لا يتغير شيء"], 1, "النهاية."),
        Q("الخسارة ميلها موجب عند وزن ما. لتقليل الخسارة…", ["زد الوزن", "قلّل الوزن", "اتركه"], 1, "عكس الميل."),
    ])
    takeaway("الميل = Δy/Δx. القاطع بين نقطتين؛ عندما تلتصقان يصبح مماسًا، وميله هو المشتقة.")
    lesson_footer(LESSON, ["معدل التغير = ميل القاطع.", "h → 0: قاطع → مماس.", "إشارة الميل تخبرك اتجاه تعديل الوزن."])
