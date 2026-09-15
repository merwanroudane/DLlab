import streamlit as st

from components.callouts import definition, intuition, takeaway
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import pipeline

LESSON = Lesson(
    id="foundations.neuron.line_to_neuron",
    title_ar="من الخط المستقيم إلى الخلية العصبية في أربع خطوات",
    title_en="From a Straight Line to a Neuron in Four Steps",
    module="foundations.neuron",
    order=1,
    prerequisites=["foundations.ml.logistic_regression", "foundations.linalg.dot_product"],
    objectives_ar=["تتبع التعميم من معادلة الخط إلى الخلية العصبية.", "قراءة رموز الاقتصاد القياسي ورموز التعلم العميق كوجهين لنفس الشيء."],
    terms=["weight", "bias", "model"],
    difficulty="beginner",
    summary_ar="β₀+β₁x → wx+b → Σwⱼxⱼ+b → f(Σwⱼxⱼ+b). أربع خطوات وكل واحدة تضيف فكرة واحدة.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("الطريق", "The path")
    pipeline(["y = β₀ + β₁x", "z = wx + b", "z = Σ wⱼxⱼ + b", "a = f(z)"], active=3)
    equation(r"\text{1)}\quad y = \beta_0 + \beta_1 x", [(r"\beta_1", "المعامل: أثر وحدة من $x$."), (r"\beta_0", "التقاطع.")],
             meaning_ar="الانحدار البسيط كما في الاقتصاد القياسي.", example_ar="الدرجة = 45 + 4.6 × ساعات.", dl_link_ar="نفس الشيء بأسماء مختلفة في الخطوة التالية.", title_ar="الخطوة 1: الخط")
    equation(r"\text{2)}\quad z = w\,x + b", [("w", "الوزن = $\\beta_1$."), ("b", "الانحياز = $\\beta_0$."), ("z", "المجموع الموزون (بدل $y$ لأنه لن يكون المخرج النهائي).")],
             meaning_ar="تغيير أسماء فقط: التعلم العميق يسمّي المعامل وزنًا والتقاطع انحيازًا.", example_ar="$z = 4.6x + 45$.", dl_link_ar="الاسم `z` سيبقى لكل «ما قبل التنشيط» في كل الطبقات.", title_ar="الخطوة 2: إعادة تسمية")
    equation(r"\text{3)}\quad z = \sum_{j=1}^{d} w_j x_j + b = \mathbf{w}\cdot\mathbf{x} + b", [("d", "عدد الخصائص."), (r"\mathbf{w}\cdot\mathbf{x}", "ضرب نقطي (درس الجبر الخطي).")],
             meaning_ar="عدة مدخلات بدل واحد: الانحدار المتعدد.", example_ar="$z = 0.5\\cdot\\text{ساعات} - 1\\cdot\\text{غياب} + 2\\cdot\\text{نوم} + 0.1$.", dl_link_ar="في الكود: `z = X @ w + b`.", title_ar="الخطوة 3: عدة مدخلات")
    equation(r"\text{4)}\quad a = f(z) = f\!\left(\sum_j w_j x_j + b\right)", [("f", "دالة التنشيط: لاخطية (Sigmoid، ReLU...)."), ("a", "التنشيط: مخرج الخلية.")],
             meaning_ar="إضافة اللاخطية. هذه **خلية عصبية** كاملة.", example_ar="$f = \\sigma$: انحدار لوجستي. $f = \\text{ReLU}$: وحدة مخفية نموذجية.",
             dl_link_ar="الشبكة = خلايا كثيرة في طبقات؛ مخرج خلية يصبح مدخلًا لأخرى.", title_ar="الخطوة 4: التنشيط")
    intuition("لم يحدث شيء «سحري» في أي خطوة: إعادة تسمية، ثم تعميم إلى عدة مدخلات، ثم دالة لاخطية. الخلية العصبية = انحدار متعدد + تنشيط.")
    compare_table(["الاقتصاد القياسي", "التعلم العميق", "الشيء نفسه"],
                  [("β (معامل)", "w (وزن)", "يضرب المدخل"), ("β₀ (تقاطع)", "b (انحياز)", "يُضاف"), ("ŷ = Xβ", "z = Xw + b", "الجزء الخطي"),
                   ("دالة الربط (logit link)", "دالة التنشيط", "اللاخطية"), ("الانحدار اللوجستي", "خلية بـ Sigmoid", "نموذج واحد"), ("المربعات الصغرى / الاحتمال الأرجح", "الانحدار التدريجي على الخسارة", "التقدير")],
                  ["rtl", "rtl", "rtl"])
    quiz("neuron.path", [
        Q("الوزن w يقابل في الانحدار…", ["β₀", "β₁", "الخطأ"], 1, "المعامل."),
        Q("ما الذي تضيفه الخطوة 4؟", ["مدخلات أكثر", "اللاخطية", "الانحياز"], 1, "دالة التنشيط."),
        Q("خلية بتنشيط Sigmoid هي…", ["انحدار خطي", "انحدار لوجستي", "شبكة عميقة"], 1, "نفس النموذج."),
    ])
    takeaway("الخلية العصبية = انحدار متعدد + دالة تنشيط. الأسماء تغيّرت، الرياضيات نفسها.")
    lesson_footer(LESSON, ["أربع خطوات: تسمية، تعميم، لاخطية.", "β ↔ w، β₀ ↔ b، link ↔ activation.", "z قبل التنشيط، a بعده."])
