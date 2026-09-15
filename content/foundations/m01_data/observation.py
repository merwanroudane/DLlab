import numpy as np
import pandas as pd
import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.data.observation",
    title_ar="الملاحظة (العينة، المثال، الصف)",
    title_en="Observation (Sample, Instance, Row)",
    module="foundations.data",
    order=3,
    prerequisites=["foundations.data.dataset"],
    objectives_ar=[
        "فهم أن الملاحظة صف واحد يمثل كيانًا واحدًا.",
        "معرفة المرادفات: `Sample`, `Instance`, `Example`, `Row`, `Record`.",
        "الوصول إلى ملاحظة واحدة ببايثون وفهم شكلها.",
        "إدراك أن الملاحظة ليست دائمًا «شخصًا»: قد تكون يومًا، أو صورة، أو جملة.",
    ],
    terms=["observation", "dataset", "shape"],
    difficulty="beginner",
    summary_ar="الملاحظة = كيان واحد بقيم كل أعمدته. لها مرادفات كثيرة ومعنى واحد.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("ما هي؟", "What is it?")
    definition(
        "**الملاحظة** `Observation` هي **صف واحد** في مجموعة البيانات: كيان واحد مع قيم جميع قياساته. "
        "تُسمى أيضًا **عينة** `Sample` و**مثالًا** `Example` و**حالة** `Instance` و**سجلًا** `Record`. "
        "كلها كلمة واحدة بمعنى واحد في التعلم الآلي."
    )
    intuition("إذا كان الجدول دفترًا، فالملاحظة هي بطاقة واحدة فيه تحمل كل ما نعرفه عن كيان واحد.")
    h2("ما الذي يمكن أن يكون ملاحظة؟", "What can an observation be?")
    table(
        ["المجال", "الملاحظة الواحدة هي…", "المدخلات", "الهدف"],
        [
            ("تمويل", "عميل واحد", "الدخل، العمر، التأخيرات", "هل تعثر؟"),
            ("اقتصاد كلي", "شهر واحد", "التضخم السابق، سعر الفائدة، البطالة", "التضخم القادم"),
            ("تسويق", "مراجعة واحدة", "كلمات النص", "إيجابية/سلبية"),
            ("رؤية حاسوبية", "صورة واحدة", "قيم البكسل", "الفئة"),
            ("سلاسل زمنية", "نافذة من 30 يومًا", "30 قيمة متتالية", "قيمة اليوم 31"),
        ],
        ["rtl", "rtl", "rtl", "rtl"],
    )
    research_note(
        "في السلاسل الزمنية الملاحظة ليست «يومًا» بالضرورة بل **نافذة** من أيام متتالية. تحديد ما هي "
        "الملاحظة هو أول قرار تصميمي في مشاريع `RNN/LSTM` وسنعود إليه في الأسبوع 10."
    )
    h2("الوصول إلى ملاحظة ببايثون", "Accessing one observation")
    X = np.array([[4200, 0, 34], [6100, 2, 45], [2900, 5, 29]], dtype=np.float32)
    st.code(
        'X = np.array([[4200, 0, 34],\n              [6100, 2, 45],\n              [2900, 5, 29]], dtype=np.float32)\n\n'
        'first = X[0]          # الملاحظة الأولى\nprint(first)          # [4200.    0.   34.]\n'
        'print(first.shape)    # (3,)  ← 3 خصائص، بلا محور ملاحظات\nprint(X[0:1].shape)   # (1, 3) ← ملاحظة واحدة مع الإبقاء على محور الملاحظات',
        language="python",
    )
    st.markdown("**المخرجات الفعلية:**")
    st.code(f"{X[0]}\n{X[0].shape}\n{X[0:1].shape}", language="text")
    common_mistake(
        "الشبكات تتوقع **دائمًا** محور الملاحظات (محور الدفعة) حتى لملاحظة واحدة. تمرير `X[0]` بالشكل `(3,)` "
        "إلى `model.predict` يعطي خطأ شكل؛ الصحيح `X[0:1]` بالشكل `(1, 3)` أو `X[0][np.newaxis, :]`."
    )
    h2("مصطلح «عينة» بين الإحصاء والتعلم الآلي", "Sample: statistics vs ML")
    table(
        ["السياق", "Sample تعني", "مثال"],
        [
            ("الإحصاء", "مجموعة من الملاحظات مسحوبة من مجتمع", "عينة من 500 أسرة"),
            ("التعلم الآلي / التعلم العميق", "ملاحظة واحدة", "`batch_size=32` = 32 عينة في الدفعة"),
        ],
        ["rtl", "rtl", "rtl"],
    )
    quiz(
        "data.observation",
        [
            Q("`X.shape == (1000, 12)`. ما شكل الملاحظة الواحدة `X[5]`؟", ["(12,)", "(1000,)", "(1, 1000)"], 0,
              "صف واحد بـ 12 خاصية.", kind="shape"),
            Q("في `Keras`، `batch_size=64` تعني 64…", ["خاصية", "ملاحظة", "حقبة"], 1, "عينة = ملاحظة في التعلم الآلي."),
            Q("لماذا يفشل `model.predict(X[0])` غالبًا؟", ["لأن القيم كبيرة", "لأن محور الملاحظات مفقود", "لأن الملاحظة الأولى خاصة"], 1,
              "الشبكة تتوقع (1, d) لا (d,).", kind="error"),
        ],
    )
    takeaway("الملاحظة = صف = كيان واحد. Sample/Instance/Example مرادفات. لا تُسقط محور الملاحظات عند التنبؤ لملاحظة واحدة.")
    lesson_footer(LESSON, [
        "الملاحظة صف واحد بقيم كل الأعمدة.",
        "قد تكون شخصًا أو شهرًا أو صورة أو نافذة زمنية.",
        "X[0] شكله (d,)، وX[0:1] شكله (1, d) وهو ما تتوقعه الشبكة.",
    ])
