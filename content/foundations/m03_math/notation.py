import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.math.notation",
    title_ar="الرموز: المتغير والثابت والمعامل والمعادلة",
    title_en="Notation: Variables, Constants, Coefficients & Equations",
    module="foundations.math",
    order=1,
    prerequisites=["foundations.start.model_training_prediction"],
    objectives_ar=[
        "قراءة معادلة رمزًا رمزًا: ما المتغير؟ ما الثابت؟ ما المعامل؟",
        "فهم الرموز الفرعية $x_i$ و$w_{ij}$ والعلوية $x^2$ والقبعة $\\hat{y}$.",
        "ترجمة معادلة إلى كود والعكس.",
    ],
    terms=["weight", "bias", "prediction"],
    difficulty="beginner",
    summary_ar="المعادلة جملة بالرموز: متغيرات تتغير، ثوابت لا، معاملات تضرب، ورموز فرعية تعدّ.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("المعادلة جملة", "An equation is a sentence")
    intuition("المعادلة ليست لغزًا بل جملة مضغوطة. $\\hat{y} = w x + b$ تقرأ: «التنبؤ يساوي الوزن ضرب المدخل زائد الانحياز». كل رمز كلمة.")
    table(
        ["المفهوم", "English", "المعنى", "في $\\hat{y} = wx + b$", "في الكود"],
        [("متغير", "Variable", "قيمة تتغير من ملاحظة لأخرى", "$x$، $\\hat{y}$", "x, y_hat"),
         ("معامل / معلمة", "Coefficient / Parameter", "قيمة تضرب متغيرًا ويتعلمها النموذج", "$w$", "w"),
         ("ثابت (إزاحة)", "Constant / Intercept", "قيمة تُضاف؛ هنا معلمة أيضًا", "$b$", "b"),
         ("رمز فرعي", "Subscript", "فهرس يعدّ: أي ملاحظة؟ أي خاصية؟", "$x_i$، $w_j$", "x[i], w[j]"),
         ("رمز علوي", "Superscript", "أس (قوة) — أو أحيانًا رقم طبقة بين قوسين", "$x^2$، $W^{(1)}$", "x**2, W1"),
         ("قبعة", "Hat", "تقدير/تنبؤ لا قيمة حقيقية", "$\\hat{y}$", "y_hat"),
         ("عريض", "Bold", "متجه أو مصفوفة (عدة أرقام)", "$\\mathbf{x}$، $\\mathbf{W}$", "x (array), W")],
        ["rtl", "ltr", "rtl", "rtl", "code"],
    )
    definition("**المتغير** يأخذ قيمًا مختلفة عبر البيانات. **المعامل** (المعلمة) يضرب متغيرًا وهو ثابت عبر البيانات لكنه يتغير أثناء **التدريب**. **الثابت** لا يتغير أبدًا ($\\pi$، 2). **المعادلة** تساوي بين طرفين.")
    h2("مثال بأكثر من مدخل", "Several inputs")
    equation(
        r"z = w_1 x_1 + w_2 x_2 + w_3 x_3 + b = \sum_{j=1}^{3} w_j x_j + b",
        [("z", "المجموع الموزون: ناتج قبل التنشيط."), ("x_j", "الخاصية رقم $j$ لملاحظة واحدة (الدخل، العمر، التأخيرات)."),
         ("w_j", "الوزن المقابل للخاصية $j$؛ إشارته اتجاه الأثر وحجمه قوته."), ("b", "الانحياز: القيمة عندما تكون كل الخصائص صفرًا."),
         (r"\sum_{j=1}^{3}", "اجمع الحدود من $j=1$ إلى $3$ — اختصار للحدود الثلاثة.")],
        meaning_ar="كل خاصية تُضرب في وزنها، تُجمع النتائج، ويُضاف الانحياز.",
        example_ar="$x = (2, 0, 1)$، $w = (0.5, -1, 2)$، $b = 0.1$: $z = 1 + 0 + 2 + 0.1 = 3.1$.",
        dl_link_ar="هذه معادلة الخلية العصبية الواحدة بالضبط. في الكود: `z = np.dot(w, x) + b`.",
        title_ar="المجموع الموزون",
    )
    st.code("import numpy as np\nx = np.array([2.0, 0.0, 1.0]); w = np.array([0.5, -1.0, 2.0]); b = 0.1\nz = np.dot(w, x) + b      # 3.1\nz = (w * x).sum() + b     # نفس الشيء، حرفيًا Σ w_j x_j", language="python")
    common_mistake("قراءة $x_2$ على أنها «$x$ تربيع». الفرعي (أسفل) فهرس؛ العلوي (أعلى) أس. $x_2^2$ = مربع الخاصية الثانية.")
    common_mistake("الخلط بين $i$ و$j$: في المنصة $i$ يعدّ الملاحظات (الصفوف) و$j$ الخصائص (الأعمدة)، و$k$ الفئات أو الوحدات.")
    quiz("math.notation", [
        Q("في $\\hat{y} = wx + b$، أيها يتغير أثناء التدريب؟", ["$x$", "$w$ و $b$", "$\\hat{y}$ فقط"], 1, "المعلمات تتعلم."),
        Q("$x_3$ تعني…", ["x أس 3", "الخاصية/العنصر الثالث", "3 ضرب x"], 1, "رمز فرعي = فهرس."),
        Q("$\\sum_{j=1}^{4} w_j x_j$ في NumPy هو…", ["w * x", "np.dot(w, x)", "w + x"], 1, "ضرب ثم جمع.", kind="code"),
    ])
    takeaway("اقرأ المعادلة كجملة: متغيرات، معاملات، ثوابت. الفرعي يعدّ، العلوي يرفع، القبعة تقدّر، Σ يجمع.")
    lesson_footer(LESSON, ["متغير يتغير بالبيانات، معلمة تتغير بالتدريب، ثابت لا يتغير.", "x_i فهرس، x² أس.", "Σ w_j x_j = np.dot(w, x)."])
