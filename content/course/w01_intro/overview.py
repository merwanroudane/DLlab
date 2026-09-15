import streamlit as st

from components.callouts import practical_note, research_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline, table

LESSON = Lesson(
    id="course.w01.overview",
    title_ar="نظرة عامة على الأسبوع وربطه بالأسس",
    title_en="Week Overview & Links to Foundations",
    module="course.w01",
    order=1,
    prerequisites=["foundations.start.ai_ml_dl", "foundations.start.model_training_prediction"],
    objectives_ar=[
        "معرفة ما يغطيه الأسبوع الأول وما يُتوقع منك في نهايته.",
        "إجراء اختبار قبلي سريع (Pre-test) لتحديد ما تحتاج مراجعته من الأسس.",
    ],
    terms=["deep_learning", "model", "training", "inference"],
    difficulty="beginner",
    summary_ar="خريطة الأسبوع الأول، الاختبار القبلي، وروابط الأسس التي يعتمد عليها.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["Intro to DL", "Core concepts", "ML vs DL", "Applications", "Post-test"], active=0)
    table(
        ["الموضوع", "الدرس في المقرر", "الدرس التأسيسي الذي يشرحه"],
        [
            ("ما التعلم العميق؟ الفرق عن التعلم الآلي", "المفاهيم الأساسية", "الأسس 0: AI vs ML vs DL"),
            ("نموذج / بنية / تدريب / استدلال", "المفاهيم الأساسية", "الأسس 0: النموذج والتدريب والتنبؤ"),
            ("لماذا بيانات + رياضيات + تحسين", "المفاهيم الأساسية", "الأسس 0: الأركان الثلاثة"),
            ("تطبيقات اقتصادية وإدارية", "التطبيقات", "الأسس 1: الهدف ونوع المسألة"),
        ],
        ["rtl", "rtl", "rtl"],
    )
    practical_note("إن لم تكن قد أنهيت الوحدة 0 من الأسس، فافعل ذلك أولًا؛ هذا الأسبوع يفترضها.")
    with st.container(horizontal=True):
        st.button("الوحدة 0 — ابدأ من هنا", icon=":material/flag:", on_click=go, args=("foundations.start",), key="w01_go_start")
        st.button("الوحدة 1 — أسس البيانات", icon=":material/table_chart:", on_click=go, args=("foundations.data",), key="w01_go_data")

    h2("الاختبار القبلي", "Pre-test")
    st.markdown("اختبار قصير **قبل** الدرس لقياس نقطة انطلاقك. لا يُحتسب؛ استخدمه لتعرف ما تراجعه.")
    quiz(
        "w01.pretest",
        [
            Q("التعلم العميق هو…", ["مرادف للذكاء الاصطناعي", "فرع من التعلم الآلي يستخدم شبكات متعددة الطبقات", "أي برنامج يستخدم بيانات"], 1,
              "راجع درس AI vs ML vs DL إن أخطأت."),
            Q("أثناء الاستدلال، المعلمات…", ["تتغير", "مجمّدة", "تُحذف"], 1, "راجع درس النموذج والتدريب."),
            Q("ما الذي يقلّله التدريب؟", ["عدد الخصائص", "الخسارة", "حجم الدفعة"], 1, "التدريب = تقليل الخسارة."),
            Q("نموذج يتنبأ بالتضخم من الفائدة بدقة عالية. هل يثبت أن الفائدة تسبب التضخم؟", ["نعم", "لا؛ التنبؤ لا يعني السببية", "فقط إذا كانت الدقة 100%"], 1,
              "نزاهة البحث: تنبؤ ≠ سببية."),
        ],
        title_ar="الاختبار القبلي — الأسبوع 01",
    )
    research_note("سنعود إلى نفس الأسئلة في الاختبار البعدي نهاية الأسبوع لقياس التقدم.")
    takeaway("الأسبوع 01 يبني اللغة المشتركة. الاختبار القبلي يوجهك إلى ما تراجعه في الأسس.")
    lesson_footer(LESSON, ["الأسبوع الأول = مفاهيم + فرق ML/DL + تطبيقات.", "أنهِ الوحدة 0 من الأسس أولًا."])
