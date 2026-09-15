import streamlit as st

from components.callouts import practical_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline, table

LESSON = Lesson(
    id="course.w02.overview",
    title_ar="نظرة عامة على الأسبوع 02 والاختبار القبلي",
    title_en="Week 02 Overview & Pre-test",
    module="course.w02",
    order=1,
    prerequisites=["course.w01.applications", "foundations.ml.task_types", "foundations.eval.splits_metrics"],
    objectives_ar=["خريطة الأسبوع وربط كل موضوع بدرسه التأسيسي.", "اختبار قبلي يوجّهك إلى ما تراجعه."],
    terms=["machine_learning", "model", "target"],
    difficulty="beginner",
    summary_ar="أنواع النماذج ← الانحدار/التصنيف ← التقييم والتقسيم والمقاييس وخطوط الأساس ← التعميم ← من ML إلى DL.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["Model types", "Regression / Classification", "Split · Metrics · Baselines", "Generalization", "ML → DL", "Post-test"], active=0)
    table(["الموضوع", "درس الأسبوع", "الأسس التي يعتمد عليها"],
          [("أنواع النماذج والمهام", "أنواع النماذج", "الأسس 7: النماذج والمهام؛ الأسس 1: الهدف"), ("الانحدار والتصنيف بمثال", "أنواع النماذج", "الأسس 7: الانحدار الخطي واللوجستي"),
           ("تدريب/تحقق/اختبار، المقاييس، خطوط الأساس", "التقييم وخطوط الأساس", "الأسس 8: التقسيم والتسريب؛ الأسس 18: المقاييس"), ("التعميم", "التقييم وخطوط الأساس", "الأسس 19"),
           ("من التعلم الآلي إلى العميق", "من ML إلى DL", "الأسس 9–10: من الانحدار إلى الخلية والشبكة")], ["rtl", "rtl", "rtl"])
    practical_note("هذا الأسبوع لا يعيد الأسس بل **يُشغّلها معًا** على مسألة واحدة (تعثر القروض) من السؤال إلى الحكم. إن كانت الوحدات 7 و8 و18 و19 غير مألوفة فراجعها أولًا.")
    with st.container(horizontal=True):
        st.button("الأسس 7 — التعلم الآلي", icon=":material/menu_book:", on_click=go, args=("foundations.ml",), key="w02_go_ml")
        st.button("الأسس 8 — إعداد البيانات", icon=":material/menu_book:", on_click=go, args=("foundations.prep",), key="w02_go_prep")
        st.button("الأسس 18 — التقييم", icon=":material/menu_book:", on_click=go, args=("foundations.eval",), key="w02_go_eval")
    h2("الاختبار القبلي", "Pre-test")
    quiz("w02.pretest", [
        Q("التنبؤ بسعر عقار بالدينار هو…", ["تصنيف", "انحدار", "تجميع"], 1, "هدف عددي متصل."),
        Q("مجموعة التحقق تُستخدم لـ…", ["تدريب المعلمات", "اختيار المعلمات الفائقة والإيقاف", "التقرير النهائي"], 1, "الاختبار للتقرير مرة واحدة."),
        Q("فئة إيجابية 3% فقط: الدقة 97% تعني…", ["نموذج ممتاز", "ربما لا شيء: التنبؤ بالفئة الغالبة يعطيها", "تسريب"], 1, "خط الأساس."),
        Q("فجوة التعميم هي الفرق بين…", ["التدريب والتحقق", "الانحدار والتصنيف", "الدقة والخسارة"], 0, "أداء التدريب مقابل المحجوز."),
        Q("متى تتفوق الشبكة العصبية على الانحدار اللوجستي عادةً؟", ["دائمًا", "عندما تكون العلاقة غير خطية والبيانات كافية", "أبدًا"], 1, "غير خطية + بيانات."),
    ], title_ar="الاختبار القبلي — الأسبوع 02")
    takeaway("الأسبوع 02 = المنهج العلمي للتعلم الآلي على مسألة واحدة: نوع المهمة، تقسيم، خط أساس، مقياس، تعميم — ثم لماذا الشبكة.")
    lesson_footer(LESSON, ["خريطة الأسبوع.", "الاختبار القبلي."])
