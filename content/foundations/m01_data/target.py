import numpy as np
import streamlit as st

from components.callouts import common_mistake, definition, practical_note, takeaway, why
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.data.target",
    title_ar="الهدف (التسمية، الاستجابة، المخرج)",
    title_en="Target (Label, Response, Output)",
    module="foundations.data",
    order=5,
    prerequisites=["foundations.data.feature"],
    objectives_ar=[
        "تعريف الهدف كالعمود الذي نريد التنبؤ به.",
        "التمييز بين هدف عددي (انحدار) وهدف فئوي (تصنيف).",
        "فهم أشكال الهدف الشائعة: `(n,)`, `(n, 1)`, `(n, k)`.",
        "ربط شكل الهدف بدالة الخسارة وطبقة الإخراج لاحقًا.",
    ],
    terms=["target", "feature", "shape"],
    related=["foundations.data.feature", "foundations.data.variable_types"],
    difficulty="beginner",
    summary_ar="الهدف ما نتنبأ به؛ نوعه (عددي/فئوي) وشكله يحددان الخسارة وطبقة الإخراج.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("ما هو؟", "What is it?")
    definition(
        "**الهدف** `Target` هو العمود الذي نريد أن يتعلم النموذج التنبؤ به. يُسمى **تسمية** `Label` في التصنيف، "
        "و**استجابة** `Response` أو **متغيرًا تابعًا** `Dependent variable` في الإحصاء، و**مخرجًا** `Output` في أطر العمل. "
        "نرمز له بـ $y$."
    )
    why("نوع الهدف هو أول قرار يحدد كل ما بعده: نوع المسألة، دالة الخسارة، طبقة الإخراج، ومقاييس التقييم.")
    h2("نوع الهدف يحدد نوع المسألة", "Target type → task type")
    compare_table(
        ["نوع الهدف", "المسألة", "مثال", "شكل y", "الخسارة لاحقًا"],
        [
            ("عددي متصل", "Regression", "سعر عقار، تضخم الشهر القادم", "(n,) أو (n, 1)", "MSE / MAE"),
            ("فئتان (0/1)", "Binary classification", "تعثر/لا تعثر", "(n,) قيم 0 أو 1", "Binary cross-entropy"),
            ("عدة فئات (واحدة لكل ملاحظة)", "Multiclass classification", "تصنيف قطاع الشركة", "(n,) أرقام الفئات أو (n, k) ترميز one-hot", "Categorical cross-entropy"),
            ("عدة تسميات معًا", "Multilabel classification", "مواضيع مقال (اقتصاد + سياسة)", "(n, k) من 0/1", "Binary cross-entropy لكل تسمية"),
        ],
        ["rtl", "ltr", "rtl", "code", "ltr"],
    )
    h2("أشكال الهدف", "Target shapes")
    y_vec = np.array([0, 1, 0, 1], dtype=np.float32)
    y_col = y_vec.reshape(-1, 1)
    y_oh = np.eye(3, dtype=np.float32)[[0, 2, 1, 0]]
    st.code(
        "y_vec = np.array([0, 1, 0, 1], dtype=np.float32)     # (4,)\n"
        "y_col = y_vec.reshape(-1, 1)                          # (4, 1)\n"
        "y_oh  = np.eye(3)[[0, 2, 1, 0]]                       # (4, 3) one-hot لثلاث فئات",
        language="python",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**`(4,)`**"); st.code(str(y_vec), language="text")
    with c2:
        st.markdown("**`(4, 1)`**"); st.code(str(y_col), language="text")
    with c3:
        st.markdown("**`(4, 3)`**"); st.code(str(y_oh), language="text")
    st.markdown(
        """
- `(n,)` متجه: قيمة واحدة لكل ملاحظة. الأكثر شيوعًا للانحدار والتصنيف الثنائي وأرقام الفئات.
- `(n, 1)` عمود واحد: نفس المعلومة كمصفوفة. بعض الأطر تفضّله لأنه يطابق شكل مخرج طبقة `Dense(1)`.
- `(n, k)` ترميز `one-hot`: عمود لكل فئة، واحد فقط يساوي 1. يُستخدم مع `categorical_crossentropy`.
"""
    )
    common_mistake(
        "هدف بأرقام الفئات `[0, 2, 1]` بالشكل `(n,)` مع خسارة `categorical_crossentropy` التي تتوقع `(n, k)`. "
        "الحل: إما `sparse_categorical_crossentropy` مع الشكل `(n,)`، أو تحويل الهدف إلى one-hot. سنعود لهذا في وحدة الخسارة."
    )
    practical_note(
        "قاعدة عملية: اكتب شكل `y` على ورقة قبل اختيار الخسارة. `(n,)` عددي ← MSE. `(n,)` من 0/1 ← BCE. "
        "`(n,)` أرقام فئات ← Sparse CCE. `(n, k)` one-hot ← CCE."
    )
    quiz(
        "data.target",
        [
            Q("هدف بقيم مثل 3.7، 12.1، 8.4. ما نوع المسألة؟", ["تصنيف ثنائي", "انحدار", "تصنيف متعدد"], 1, "قيم عددية متصلة."),
            Q("ما شكل `np.eye(4)[[1, 3]]`؟", ["(2, 4)", "(4, 2)", "(2,)"], 0, "ملاحظتان، 4 فئات one-hot.", kind="shape"),
            Q("هدف بأرقام الفئات بالشكل `(n,)`. أي خسارة تناسبه مباشرة؟", ["categorical_crossentropy", "sparse_categorical_crossentropy", "mse"], 1,
              "Sparse تتعامل مع أرقام الفئات دون one-hot.", kind="code"),
        ],
    )
    takeaway("الهدف ما نتنبأ به. نوعه يحدد المسألة، وشكله يحدد الخسارة وطبقة الإخراج.")
    lesson_footer(LESSON, [
        "Target = Label = Response = Output = y.",
        "عددي ← انحدار؛ فئوي ← تصنيف (ثنائي/متعدد/متعدد التسميات).",
        "الأشكال (n,)، (n, 1)، (n, k) ليست تفاصيل: كل شكل له خسارة مناسبة.",
    ])
