import streamlit as st

from components.callouts import practical_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go
from core.rtl import table

LESSON = Lesson(
    id="foundations.data.dataset_anatomy",
    title_ar="تشريح مجموعة بيانات كاملة (تمرين ختامي)",
    title_en="Dataset Anatomy — Capstone Exercise",
    module="foundations.data",
    order=10,
    prerequisites=["foundations.data.shape_axis_rank", "foundations.data.variable_types"],
    objectives_ar=[
        "تطبيق كل مفاهيم الوحدة على مجموعة بيانات واحدة في المعمل.",
        "إنتاج وصف كامل: n، d، نوع كل عمود، الهدف، أشكال X و y، وأنواع التخزين.",
    ],
    terms=["dataset", "feature", "target", "shape", "dtype"],
    labs=["labs.dataset_anatomy"],
    difficulty="beginner",
    summary_ar="تمرين ختامي يجمع الوحدة: افتح المعمل، اختر الهدف والخصائص، واقرأ الأشكال والأنواع.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("قائمة التشريح", "The anatomy checklist")
    st.markdown("قبل أي نموذج، يجب أن تستطيع ملء هذا الجدول لأي مجموعة بيانات:")
    table(
        ["السؤال", "الرمز / الأمر", "لماذا يهم"],
        [
            ("كم ملاحظة؟", "n = X.shape[0]", "يحدد حجم النموذج الممكن وعدد الخطوات في الحقبة"),
            ("كم خاصية؟", "d = X.shape[1]", "يحدد `input_shape` لأول طبقة"),
            ("ما الهدف ونوعه؟", "y, y.dtype, np.unique(y)", "يحدد المسألة والخسارة وطبقة الإخراج"),
            ("ما نوع كل عمود إحصائيًا؟", "عددي / فئوي (ثنائي/اسمي/ترتيبي)", "يحدد الترميز والتحجيم"),
            ("ما نوع التخزين؟", "df.dtypes", "object يجب ترميزه؛ int64 ← float32"),
            ("هل الأشكال متوافقة؟", "X.shape[0] == y.shape[0]", "أول شرط لأي تدريب"),
            ("هل توجد قيم مفقودة؟", "df.isna().sum()", "الشبكة لا تقبل NaN (وحدة إعداد البيانات)"),
        ],
        ["rtl", "code", "rtl"],
    )
    h2("التمرين", "Exercise")
    st.markdown(
        """
1. افتح **معمل تشريح مجموعة البيانات**.
2. اختر مجموعة البيانات «تعثر القروض».
3. اجعل `defaulted` هدفًا، واختر ثلاث خصائص عددية.
4. اقرأ `X.shape` و`y.shape` و`dtype` وسجّلها.
5. غيّر الهدف إلى `income` وأجب: ما نوع المسألة الآن؟ وماذا تغيّر في شكل `y`؟
"""
    )
    st.button("افتح معمل تشريح مجموعة البيانات", type="primary", icon=":material/science:", on_click=go,
              args=("labs.dataset_anatomy",), key="anatomy_open_lab")
    practical_note("الحل: مع `defaulted` المسألة تصنيف ثنائي و`y` من 0/1. مع `income` المسألة انحدار و`y` قيم متصلة؛ الشكل `(n,)` في الحالتين.")
    quiz(
        "data.anatomy",
        [
            Q("أول شرط يجب التحقق منه قبل التدريب؟", ["X.shape[1] == y.shape[0]", "X.shape[0] == y.shape[0]", "X.dtype == y.dtype"], 1,
              "عدد الملاحظات في X يساوي طول y.", kind="code"),
            Q("عمود `object` في `df.dtypes` يعني…", ["أرقام كبيرة", "غالبًا نص يحتاج ترميزًا", "قيم مفقودة"], 1, "النص لا يدخل الشبكة."),
        ],
    )
    takeaway("سبعة أسئلة تجيب عنها قبل أي نموذج. إن استطعت ملء الجدول فقد أنهيت الوحدة الأولى فعلًا.")
    lesson_footer(LESSON, [
        "n, d, نوع الهدف، نوع كل عمود، dtype، توافق الأشكال، القيم المفقودة.",
        "المعمل هو المكان الذي تثبت فيه أنك تستطيع ذلك على بيانات حقيقية.",
    ])
