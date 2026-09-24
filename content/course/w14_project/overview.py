import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, practical_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from content.course.w14_project._viz import roadmap_svg
from content.course.w14_project.project_guide import STAGES
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline, table

LESSON = Lesson(
    id="course.w14.overview",
    title_ar="نظرة عامة على الأسبوع 14 والاختبار القبلي",
    title_en="Week 14 Overview & Pre-test",
    module="course.w14",
    order=1,
    prerequisites=["course.w13.colab_runtime", "course.w07.diagnose_interpret"],
    objectives_ar=["خريطة المشروع بمراحله السبع عشرة (§81) بتحريك سريع.", "اختبار قبلي على منهج المشروع."],
    terms=["baseline", "reproducibility", "data_leakage"],
    difficulty="intermediate",
    summary_ar="دليل المراحل السبع عشرة ← اختيار نوع المشروع بحسب البيانات ← قالب الدفتر ونقاط الاستبدال.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["Choose a problem", "17-stage guide", "Notebook template", "Checklist", "Post-test"], active=0)
    h2("المشروع بلمحة", "The project at a glance")
    shown = [0, 5, 6, 9, 11, 13, 16]
    caps = ["**التخطيط يبدأ بالسؤال** لا بالنموذج.", "**فحص التسريب** يختم مرحلة التخطيط: خلط الهدف يجب أن يُسقط الأداء.", "**البناء يبدأ بخط الأساس**.", "**التدريب** بإيقاف مبكر وحفظ نقاط.", "**التشخيص** بالأدلة من المنحنيات والفجوة.", "**التقييم** مرة واحدة مقابل خط الأساس.", "**العرض** يُظهر الأدلة ونقاط الضعف."]
    animation_player("w14_teaser", [Frame(roadmap_svg(STAGES, k), caption(c), action=f"stage {k + 1}") for k, c in zip(shown, caps)], title_ar="سبع محطات من سبع عشرة", interval_ms=1800)
    table(["الموضوع", "درس الأسبوع", "المرجع"],
          [("المراحل السبع عشرة وقائمة الفحص", "دليل المشروع", "الأسابيع 01–13"), ("اختيار نوع المشروع", "دليل المشروع", "الأسابيع 02–12"), ("الدفتر ونقاط الاستبدال", "قالب الدفتر", "الأسبوع 07؛ الأسبوع 13 (Colab)")],
          ["rtl", "rtl", "rtl"])
    practical_note("المشروع يُقيَّم بالمنهج لا بالدقة. اقرأ الرمز النهائي (الأسبوع 15) قبل أن تبدأ.")
    with st.container(horizontal=True):
        st.button("دليل المشروع", icon=":material/checklist:", on_click=go, args=("course.w14.project_guide",), key="w14_ov_guide")
        st.button("قالب الدفتر", icon=":material/description:", on_click=go, args=("course.w14.notebook_template",), key="w14_ov_tpl")
        st.button("الرمز النهائي", icon=":material/grading:", on_click=go, args=("course.w15.rubric",), key="w14_ov_rubric")
    intuition("كل ما تعلمته في 13 أسبوعًا يظهر هنا مرة واحدة، بالترتيب. المشروع ليس امتحانًا جديدًا بل تطبيق للمنهج نفسه على بياناتك.")
    h2("الاختبار القبلي", "Pre-test")
    quiz("course.w14.pretest", [
        Q("المرحلة الأولى في المشروع:", ["اختيار البنية", "تعريف المسألة والقرار", "التدريب"], 1, ""),
        Q("متى يُحسب خط الأساس؟", ["بعد النموذج", "قبل النموذج", "لا حاجة"], 1, ""),
        Q("فحص التسريب البسيط:", ["زيادة الحقب", "خلط الهدف يجب أن يُسقط الأداء إلى خط الأساس", "حذف الخصائص"], 1, ""),
        Q("قبل التسليم:", ["احفظ فقط", "Restart and run all ثم احفظ المخرجات", "احذف المخرجات"], 1, ""),
    ], title_ar="الاختبار القبلي — الأسبوع 14")
    takeaway("الأسبوع 14 = المشروع كسلسلة قرارات مبرَّرة في 17 مرحلة، وقالب دفتر يمنع الأخطاء الشائعة بالبنية.")
    lesson_footer(LESSON, ["المشروع بلمحة (تحريك).", "خريطة الأسبوع.", "الاختبار القبلي."])
