import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway, why
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.data.feature",
    title_ar="الخاصية (المتغير المستقل، المدخل)",
    title_en="Feature (Predictor, Variable, Input)",
    module="foundations.data",
    order=4,
    prerequisites=["foundations.data.observation"],
    objectives_ar=[
        "تعريف الخاصية كعمود مدخل يصف الملاحظة.",
        "ربط المرادفات: `Feature`, `Predictor`, `Independent variable`, `Input`, `Column`, `Attribute`.",
        "فهم أن الخاصية قد تكون خامًا أو مشتقة (هندسة الخصائص).",
        "التمييز بين خاصية وهدف بحسب سؤال البحث لا بحسب العمود.",
    ],
    terms=["feature", "target", "observation"],
    related=["foundations.data.target"],
    difficulty="beginner",
    summary_ar="الخاصية عمود مدخل يصف الملاحظة؛ نفس العمود قد يكون خاصية في سؤال وهدفًا في سؤال آخر.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("ما هي؟", "What is it?")
    definition(
        "**الخاصية** `Feature` هي **عمود مدخل** يصف الملاحظة ويستخدمه النموذج للتنبؤ. في الاقتصاد القياسي "
        "تُسمى **متغيرًا مستقلًا** أو **متغيرًا تفسيريًا**، وفي قواعد البيانات **سمة** `Attribute`، "
        "وفي أطر التعلم العميق ببساطة **مدخلًا** `Input`. عدد الخصائص نرمز له بـ $d$."
    )
    intuition("الخصائص هي «الأدلة» التي يملكها المحقق. كلما كانت الأدلة ذات صلة قلّ اعتماد النموذج على التخمين.")
    h2("المرادفات في السياقات المختلفة", "Synonyms across fields")
    compare_table(
        ["السياق", "المصطلح", "ملاحظة"],
        [
            ("التعلم الآلي", "Feature", "الأكثر شيوعًا"),
            ("الاقتصاد القياسي / الإحصاء", "Independent / explanatory variable, regressor", "نفس المعنى، مع تركيز على التفسير"),
            ("التعلم العميق / أطر العمل", "Input, input dimension", "`input_shape=(d,)` في `Keras`"),
            ("قواعد البيانات", "Column, attribute, field", "بنية التخزين"),
        ],
        ["rtl", "ltr", "rtl"],
    )
    h2("خام أم مشتقة؟", "Raw vs engineered")
    st.markdown(
        """
- **خاصية خام**: قياس مباشر، مثل `income`.
- **خاصية مشتقة**: نحسبها من خصائص أخرى، مثل `debt_to_income = debt / income`. هذا ما يسمى **هندسة الخصائص** `Feature Engineering`.

الفرق الجوهري بين التعلم الآلي التقليدي والتعلم العميق: في الأول يهندس الباحث الخصائص يدويًا، وفي الثاني
تتعلم الطبقات الأولى **تمثيلات** `Representations` تقوم بدور الخصائص المشتقة تلقائيًا. لكن حتى مع الشبكات
العميقة، اختيار الخصائص الخام الصحيحة يبقى مسؤولية الباحث.
"""
    )
    h2("خاصية أم هدف؟ سؤال البحث يقرر", "Feature or target? The question decides")
    why(
        "العمود `income` هو **خاصية** إذا كان السؤال «هل سيتعثر العميل؟»، وهو **هدف** إذا كان السؤال "
        "«ما الدخل المتوقع للعميل من بقية بياناته؟». لا يوجد شيء اسمه عمود «خاصية بطبيعته»."
    )
    common_mistake(
        "إدخال الهدف نفسه (أو نسخة مقنّعة منه) ضمن الخصائص، مثل استخدام `total_due_after_default` للتنبؤ "
        "بـ `defaulted`. هذا **تسريب هدف** `Target Leakage`: دقة خيالية في التدريب وفشل تام في الواقع."
    )
    research_note(
        "في البحوث الاقتصادية اسأل دائمًا: هل هذه الخاصية **متاحة وقت التنبؤ**؟ خاصية تُعرف بعد وقوع الحدث "
        "لا يجوز استخدامها للتنبؤ به، حتى لو رفعت الدقة."
    )
    quiz(
        "data.feature",
        [
            Q("ما عدد الخصائص إذا كان `X.shape == (500, 7)`؟", ["500", "7", "3500"], 1, "الرقم الثاني = d.", kind="shape"),
            Q("العمود `income` هو…", ["خاصية دائمًا", "هدف دائمًا", "خاصية أو هدف حسب سؤال البحث"], 2, "الدور يحدده السؤال."),
            Q("استخدام خاصية تُعرف بعد وقوع الحدث للتنبؤ به يسمى…", ["هندسة خصائص", "تسريب هدف", "تعميم"], 1, "Target leakage."),
        ],
    )
    takeaway("الخاصية عمود مدخل؛ المرادفات كثيرة والمعنى واحد. الدور (خاصية/هدف) يحدده سؤال البحث، والتسريب أخطر خطأ.")
    lesson_footer(LESSON, [
        "Feature = Predictor = Independent variable = Input.",
        "عدد الخصائص d هو البعد الثاني في X.",
        "خصائص خام ومشتقة؛ التعلم العميق يتعلم المشتقة تلقائيًا.",
        "احذر تسريب الهدف: لا خصائص تُعرف بعد الحدث.",
    ])
