import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, practical_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from content.course.w14_project._viz import slide_svg
from content.course.w15_presentation.presentation_discussion import SLIDES
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline, table

LESSON = Lesson(
    id="course.w15.overview",
    title_ar="نظرة عامة على الأسبوع 15 والاختبار القبلي",
    title_en="Week 15 Overview & Pre-test",
    module="course.w15",
    order=1,
    prerequisites=["course.w14.notebook_template"],
    objectives_ar=["رؤية العرض كقصة من عشر لقطات (تحريك سريع).", "خريطة الأسبوع الأخير.", "اختبار قبلي على العرض والتقييم."],
    terms=["baseline", "reproducibility"],
    difficulty="intermediate",
    summary_ar="هيكل العرض في 10 شرائح ← تحليل النتائج ونقاط الضعف ← المناقشة والأسئلة العشرة ← الرمز النهائي والتقييم الذاتي.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["10-slide structure", "Results & weaknesses", "Discussion", "Improvements", "Rubric & self-assessment", "Final post-test"], active=0)
    h2("العرض في لمحة", "The presentation at a glance")
    shown = [0, 3, 6, 8, 9]
    caps = ["يبدأ العرض بالسؤال والقرار — لا بالنموذج.", "خط الأساس قبل النموذج، دائمًا.", "التقييم: مرة واحدة، بوحدة الهدف، مقابل خط الأساس.", "نقاط الضعف تُعلن قبل أن تُسأل عنها.", "الاستنساخ يختم العرض: من يعيد أرقامك؟"]
    animation_player("w15_teaser", [Frame(slide_svg(SLIDES, k), caption(c), action=f"slide {k + 1}") for k, c in zip(shown, caps)], title_ar="خمس لقطات من عشر", interval_ms=1800)
    table(["الموضوع", "درس الأسبوع", "المرجع"],
          [("هيكل العرض والمناقشة والمقترحات", "العرض والمناقشة", "الأسبوع 14: المراحل"), ("الرمز النهائي والتقييم الذاتي", "الرمز", "الأسبوع 14: القالب")],
          ["rtl", "rtl", "rtl"])
    practical_note("قيّم نفسك بالرمز قبل أسبوع من التسليم لا في ليلته: البنود الأعلى وزنًا (التسريب، خط الأساس، التقييم) هي الأرخص وقتًا.")
    with st.container(horizontal=True):
        st.button("العرض والمناقشة", icon=":material/slideshow:", on_click=go, args=("course.w15.presentation_discussion",), key="w15_ov_pres")
        st.button("الرمز النهائي", icon=":material/grading:", on_click=go, args=("course.w15.rubric",), key="w15_ov_rubric")
    intuition("العرض الجيد يجيب عن سؤال واحد في كل شريحة، والمناقشة الجيدة تبدأ بما تعرفه عن ضعفك.")
    h2("الاختبار القبلي", "Pre-test")
    quiz("course.w15.pretest", [
        Q("أهم ما يُعرض قبل دقة النموذج:", ["عدد الطبقات", "خط الأساس", "زمن التدريب"], 1, ""),
        Q("سؤال «هل العلاقة سببية؟» جوابه النزيه غالبًا:", ["نعم", "لا: تنبؤ وترابط", "لا أعرف"], 1, ""),
        Q("نقاط الضعف في العرض…", ["تُخفى", "تُعلن بأدلة مع مقترحات", "تُذكر إن سُئلت فقط"], 1, ""),
        Q("البند الأعلى وزنًا بين هذه في الرمز:", ["النموذج", "التقييم", "الألوان"], 1, ""),
    ], title_ar="الاختبار القبلي — الأسبوع 15")
    takeaway("الأسبوع 15 = عرض بالأدلة، مناقشة بنزاهة، وتقييم ذاتي بالرمز المعلن.")
    lesson_footer(LESSON, ["العرض في لمحة (تحريك).", "خريطة الأسبوع.", "الاختبار القبلي."])
