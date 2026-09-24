import streamlit as st

from components import svgkit as K
from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, practical_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline, table

LESSON = Lesson(
    id="course.w12.overview",
    title_ar="نظرة عامة على الأسبوع 12 والاختبار القبلي",
    title_en="Week 12 Overview & Pre-test",
    module="course.w12",
    order=1,
    prerequisites=["course.w11.train_eval_diagnose"],
    objectives_ar=["رؤية عائلة الخلايا التكرارية الثلاث جنبًا إلى جنب (تحريك).", "خريطة الأسبوع.", "اختبار قبلي على GRU والمقارنة."],
    terms=["sequence", "gru", "lstm", "rnn"],
    difficulty="intermediate",
    summary_ar="بوابتا GRU ومعادلاتها ← مقارنة RNN/LSTM/GRU بإنصاف ← أنماط التطبيقات الأربعة والتحفظات العلمية.",
)

_FAM = [
    ("SimpleRNN", "**SimpleRNN**: حالة واحدة، بلا بوابات: `h = tanh(xWx + hWh + b)`. بسيطة وسريعة وقصيرة الذاكرة.", K.BLUE, "1 block"),
    ("LSTM", "**LSTM**: حالتان (c وh) وثلاث بوابات (f، i، o). ذاكرة طويلة بأربع كتل أوزان.", K.PINK, "4 blocks"),
    ("GRU", "**GRU**: حالة واحدة وبوابتان (z، r). تدمج النسيان والإدخال في z — ذاكرة طويلة بثلاث كتل فقط.", K.EMERALD, "3 blocks"),
    ("choose", "**الاختيار**: الأبسط الذي يهزم الساذج على اختبار زمني، ثم GRU، ثم LSTM إن تحسّن التحقق بفارق حقيقي.", K.VIOLET, "evidence"),
]


def _svg(k: int) -> str:
    s = K.svg_open(700, 130)
    for i, (n, _, col, sub) in enumerate(_FAM):
        x = 15 + i * 170
        s += K.box(x, 35, 150, 56, n, col, filled=i == k, size=13, sub=sub)
        if i < len(_FAM) - 1:
            s += K.arrow(x + 151, 63, x + 169, 63, color=col)
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["Update & reset gates", "GRU equations", "RNN vs LSTM vs GRU", "Applications & patterns", "Post-test"], active=0)
    h2("العائلة التكرارية", "The recurrent family")
    animation_player("w12_family", [Frame(_svg(i), caption(t), action=n) for i, (n, t, _, _) in enumerate(_FAM)], title_ar="ثلاث خلايا، سؤال واحد: أيها يكفي؟", interval_ms=2400)
    table(["الموضوع", "درس الأسبوع", "الأسس التي يعتمد عليها"],
          [("بوابتا التحديث وإعادة الضبط والمعادلات", "بوابتا GRU", "الأسبوع 11: البوابات"), ("المقارنة المنصفة", "المقارنة", "الأسبوعان 10–11: خط الأنابيب؛ الأسبوع 07: الاستقرار"), ("أنماط التطبيقات", "التطبيقات", "الأسبوع 01: من السؤال إلى المسألة")],
          ["rtl", "rtl", "rtl"])
    st.markdown(
        "**في نهاية الأسبوع ستستطيع:**\n"
        "- أن تكتب معادلات GRU وتشرح z وr بقيم حقيقية.\n"
        "- أن تحسب معلمات أي خلية تكرارية وتقارن الثلاث بإنصاف (بذور متعددة).\n"
        "- أن تحدد نمط أي تطبيق تسلسلي ورأسه وخسارته وخط أساسه."
    )
    practical_note("هذا الأسبوع يختم التسلسلات: بعده مشروعك (الأسبوع 14) يمكن أن يكون تسلسليًا بثقة.")
    with st.container(horizontal=True):
        st.button("معمل بوابات GRU", icon=":material/science:", on_click=go, args=("labs.gru_gates_lab",), key="w12_go_lab")
    intuition("كل خلية تكرارية هي نفس الحلقة عبر الزمن؛ الفرق في «كم مفتاحًا» يتحكم في الذاكرة. مفاتيح أكثر = مرونة أكثر = بيانات أكثر مطلوبة.")
    h2("الاختبار القبلي", "Pre-test")
    quiz("course.w12.pretest", [
        Q("GRU لها…", ["ثلاث بوابات وحالتان", "بوابتان وحالة واحدة", "بوابة واحدة"], 1, ""),
        Q("GRU(16) على D=1 مقابل LSTM(16):", ["أكثر معلمات", "أقل (3/4)", "نفسها"], 1, "3 كتل مقابل 4."),
        Q("المقارنة المنصفة بين نموذجين تحتاج…", ["تشغيلًا واحدًا", "نفس البيانات والتقسيم وعدة بذور", "دقة تدريب"], 1, ""),
        Q("تصنيف مشاعر مراجعة نصية نمط…", ["كثير → واحد (فئة)", "كثير → كثير", "واحد → كثير"], 0, ""),
    ], title_ar="الاختبار القبلي — الأسبوع 12")
    takeaway("الأسبوع 12 = GRU كبديل أخف لـ LSTM، مقارنة منصفة بين الثلاث، وأنماط التطبيقات.")
    lesson_footer(LESSON, ["العائلة التكرارية (تحريك).", "خريطة الأسبوع.", "الاختبار القبلي."])
