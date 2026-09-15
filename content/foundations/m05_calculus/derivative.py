import streamlit as st

from components.callouts import common_mistake, definition, intuition, practical_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.calculus.derivative",
    title_ar="المشتقة",
    title_en="The Derivative",
    module="foundations.calculus",
    order=2,
    prerequisites=["foundations.calculus.change_slope"],
    objectives_ar=[
        "تعريف المشتقة كنهاية ميل القاطع، وقراءة الرموز $f'(x)$ و$\\frac{df}{dx}$.",
        "حفظ القواعد الأساسية الخمس التي تظهر في الشبكات: القوة، الثابت، المجموع، الأس، اللوغاريتم.",
        "التحقق من أي مشتقة عدديًا بالفروق المنتهية.",
    ],
    terms=["gradient", "loss"],
    labs=["labs.derivative_lab"],
    difficulty="beginner",
    summary_ar="المشتقة ميل المماس؛ خمس قواعد تكفي للشبكات؛ الفروق المنتهية تتحقق من أي مشتقة.",
)

CODE = '''import numpy as np

def numerical_derivative(f, x, h=1e-5):
    """الفروق المنتهية المركزية: (f(x+h) - f(x-h)) / 2h"""
    return (f(x + h) - f(x - h)) / (2 * h)

rules = {{
    "x^2      -> 2x":        (lambda x: x**2,        lambda x: 2*x),
    "x^3      -> 3x^2":      (lambda x: x**3,        lambda x: 3*x**2),
    "e^x      -> e^x":       (np.exp,                np.exp),
    "ln(x)    -> 1/x":       (np.log,                lambda x: 1/x),
    "3x+5     -> 3":         (lambda x: 3*x+5,       lambda x: 3.0),
    "sigmoid  -> s(1-s)":    (lambda x: 1/(1+np.exp(-x)), lambda x: (1/(1+np.exp(-x)))*(1-1/(1+np.exp(-x)))),
    "(x-2)^2  -> 2(x-2)":    (lambda x: (x-2)**2,    lambda x: 2*(x-2)),
}}
x0 = {x0}
print(f"x0 = {{x0}}")
for name, (f, df) in rules.items():
    num, sym = numerical_derivative(f, x0), df(x0)
    print(f"{{name:<22}} numeric={{num:9.5f}}  symbolic={{sym:9.5f}}  match={{np.isclose(num, sym)}}")'''


def _controls() -> dict:
    return {"x0": st.slider("x₀", 0.5, 4.0, 1.5, 0.25, key="ctrl_deriv_x0")}


def render() -> None:
    lesson_header(LESSON)
    h2("التعريف", "Definition")
    equation(r"f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}",
             [("f'(x)", "المشتقة عند $x$ (تُقرأ «f شرطة»)؛ ميل المماس."), (r"\frac{df}{dx}", "نفس الشيء برمز لايبنتز: «تغير f بالنسبة لتغير x»."), (r"\lim_{h\to 0}", "النهاية: اجعل $h$ صغيرًا بلا حد.")],
             meaning_ar="المشتقة دالة جديدة تعطي ميل الأصلية عند كل نقطة.",
             example_ar="$f(x) = 0.5x^2$ ⟹ $f'(x) = x$؛ عند $x = 3$ الميل 3.",
             dl_link_ar="$\\frac{dL}{dw}$: كم تتغير الخسارة لكل وحدة تغير في الوزن. الأطر تحسبها تلقائيًا (autodiff) لكن فهم معناها ضروري للتشخيص.", title_ar="المشتقة")
    intuition("المشتقة «عداد سرعة» الدالة: إن كانت $f$ المسافة فالمشتقة السرعة. موجبة = تصعد، سالبة = تهبط، صفر = قمة أو قاع أو سطح مستوٍ.")
    h2("القواعد التي تكفيك", "The rules you need")
    table(["الدالة", "المشتقة", "أين تظهر في الشبكة"],
          [("c (ثابت)", "0", "الثوابت تختفي"), ("x^n", "n·x^(n−1)", "MSE: (ŷ−y)² → 2(ŷ−y)"), ("a·x + b", "a", "الطبقة الخطية: ∂z/∂x = w"),
           ("f + g", "f' + g'", "المجموع الموزون"), ("e^x", "e^x", "Softmax، sigmoid"), ("ln x", "1/x", "الإنتروبيا المتقاطعة"),
           ("σ(x) = 1/(1+e^(−x))", "σ(x)(1−σ(x))", "Sigmoid: صغيرة عند الإشباع → تلاشي التدرج"), ("max(0, x)", "1 إن x>0 وإلا 0", "ReLU: تدرج ثابت أو صفر (Dead ReLU)")],
          ["code", "code", "rtl"])
    practical_note("لن تشتق يدويًا في العمل — الأطر تفعل. لكنك ستقرأ «مشتقة Sigmoid قصوى 0.25» أو «مشتقة ReLU صفر للسالب» وتفهم فورًا لماذا تتلاشى التدرجات أو تموت الوحدات.")
    code_lab(CodeLab(
        key="calc_deriv", title_ar="تحقق عددي من المشتقات", code=CODE, template=True, defaults={"x0": 1.5},
        before=Before(goal_ar="مقارنة المشتقة الرمزية (من الجدول) بالمشتقة العددية (الفروق المنتهية) لسبع دوال.", stage_ar="تفاضل ← أدوات التحقق.",
                      inputs_ar="نقطة `x0` ودوال صغيرة.", expected_ar="سبعة أسطر بقيمتين متطابقتين تقريبًا و`match=True`.",
                      math_ar="$f'(x) \\approx \\frac{f(x+h) - f(x-h)}{2h}$ مع $h = 10^{-5}$."),
        explain=[("3-5", "الفروق المركزية أدق من الأمامية بنفس `h`. هذه الدالة أداة تشخيص عامة: تتحقق من أي تدرج."),
                 ("7-15", "زوج (الدالة، مشتقتها الرمزية) لكل قاعدة. لاحظ مشتقة Sigmoid مكتوبة بدلالة Sigmoid نفسها."),
                 ("18-20", "لكل قاعدة: احسب عدديًا ورمزيًا وقارن. `isclose` يتسامح مع فرق التقريب.")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- التطابق يؤكد الجدول. غيّر `x0` ولاحظ أن مشتقة `3x+5` ثابتة (3) بينما مشتقة `x²` تتغير (2x).\n- مشتقة Sigmoid عند `x0 = 4` صغيرة جدًا (~0.018): هذا هو الإشباع.",
    ))
    common_mistake("«المشتقة رقم». لا: المشتقة **دالة** تعطي رقمًا عند كل نقطة. $f'(x) = 2x$ دالة؛ $f'(3) = 6$ رقم.")
    quiz("calc.derivative", [
        Q("مشتقة $x^3$ هي…", ["$3x^2$", "$x^2$", "$3x$"], 0, "قاعدة القوة.", kind="equation"),
        Q("مشتقة $(w - 2)^2$ بالنسبة لـ $w$…", ["$2(w-2)$", "$2w$", "$w - 2$"], 0, "قوة + سلسلة بسيطة.", kind="equation"),
        Q("مشتقة ReLU للمدخل السالب…", ["1", "0", "غير معرفة"], 1, "الجزء المسطح."),
        Q("أقصى قيمة لمشتقة Sigmoid…", ["1", "0.25", "0.5"], 1, "عند x = 0: 0.5 × 0.5."),
    ])
    takeaway("المشتقة دالة الميل. خمس قواعد + الفروق المنتهية للتحقق. مشتقات التنشيط تفسر تلاشي التدرج وموت ReLU.")
    lesson_footer(LESSON, ["f'(x) = lim (f(x+h) − f(x))/h.", "قواعد: قوة، ثابت، مجموع، e^x، ln.", "الفروق المنتهية أداة تشخيص لأي تدرج."])
