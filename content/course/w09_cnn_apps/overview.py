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
    id="course.w09.overview",
    title_ar="نظرة عامة على الأسبوع 09 والاختبار القبلي",
    title_en="Week 09 Overview & Pre-test",
    module="course.w09",
    order=1,
    prerequisites=["course.w08.layers_shapes", "course.w07.diagnose_interpret"],
    objectives_ar=["رؤية مشروع صور كاملًا في سبع محطات (تحريك).", "خريطة الأسبوع وروابط الأسس.", "اختبار قبلي على تطبيقات CNN."],
    terms=["cnn", "data_augmentation", "feature_map"],
    difficulty="intermediate",
    summary_ar="المعالجة المسبقة ← بناء CNN وتدريبها وتقييمها ← فرط التخصيص وزيادة البيانات ← خرائط الخصائص وحساسية الإخفاء وتصحيح أخطاء الأشكال.",
)

_STEPS = [
    ("prepare", "**تجهيز الصور**: [0, 1]، بُعد القناة، دفعة رتبة 4، هدف بأعداد صحيحة.", K.BLUE),
    ("build", "**بناء CNN**: كتلتان Conv/Pool ثم رأس كثيف وsoftmax — بالأسئلة الأحد عشر.", K.VIOLET),
    ("train", "**التدريب والتقييم**: منحنيات، دقة الاختبار، مصفوفة التباس.", K.EMERALD),
    ("inside", "**داخل الشبكة**: صورة واحدة طبقةً طبقة حتى الاحتمالات.", K.CYAN),
    ("overfit", "**بيانات قليلة**: 60 صورة ⇒ حفظ. نقيس الفجوة.", K.RED),
    ("augment", "**زيادة البيانات**: تحويلات تحفظ الوسم على التدريب فقط.", K.AMBER),
    ("explain", "**التفسير والتصحيح**: نوى وخرائط، حساسية الإخفاء، وأخطاء الأشكال الخمسة.", K.PINK),
]


def _svg(k: int) -> str:
    s = K.svg_open(700, 130)
    for i, (n, _, col) in enumerate(_STEPS):
        x = 8 + i * 98
        s += K.box(x, 40, 88, 46, n, col, filled=i == k, size=12)
        if i < len(_STEPS) - 1:
            s += K.arrow(x + 89, 63, x + 97, 63, color=col)
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["Preprocess images", "Build CNN", "Train & evaluate", "Overfitting & augmentation", "Feature maps & occlusion", "Shape debugging", "Post-test"], active=0)
    h2("مشروع صور في سبع محطات", "An image project in seven stops")
    animation_player("w09_journey", [Frame(_svg(i), caption(t), action=n) for i, (n, t, _) in enumerate(_STEPS)], title_ar="من الصور الخام إلى نموذج مفسَّر", interval_ms=2200)
    table(["الموضوع", "درس الأسبوع", "الأسس التي يعتمد عليها"],
          [("معالجة الصور مسبقًا وبناء CNN وتدريبها وتقييمها", "البناء والتدريب", "الأسبوع 08؛ الأسبوع 03 (سير عمل Keras)"), ("فرط التخصيص وزيادة البيانات", "فرط التخصيص والزيادة", "الأسس 19–20"),
           ("خرائط الخصائص وحساسية الإخفاء وأخطاء الأشكال", "خرائط الخصائص والتصحيح", "الأسبوع 08؛ الوحدة 21 (الأخطاء)")],
          ["rtl", "rtl", "rtl"])
    st.markdown(
        "**في نهاية الأسبوع ستستطيع:**\n"
        "- أن تجهّز صورًا لـ CNN وتبني وتدرّب وتقيّم نموذجًا صغيرًا.\n"
        "- أن تتعرف على فرط التخصيص في الصور وتعالجه بزيادة بيانات تحفظ الوسم.\n"
        "- أن تفسّر ما تعلّمه النموذج بالخرائط وحساسية الإخفاء.\n"
        "- أن تصحح أخطاء الأشكال الشائعة من رسائلها."
    )
    practical_note("الصور هنا تركيبية صغيرة (16×16) لتتدرب في ثوانٍ، لكن كل خطوة هي نفسها في صور حقيقية كبيرة: المنهج واحد، الحجم فقط يتغير.")
    with st.container(horizontal=True):
        st.button("حاسبة أشكال CNN", icon=":material/science:", on_click=go, args=("labs.cnn_shape_calculator",), key="w09_go_calc")
        st.button("معمل فرط التخصيص", icon=":material/science:", on_click=go, args=("labs.overfitting_lab",), key="w09_go_of")
    intuition("أسبوع 08 كان «كيف يعمل الالتفاف»؛ أسبوع 09 هو «كيف تبني وتدرّب وتثق بنموذج صور». الفرق بين الفهم والممارسة.")
    h2("الاختبار القبلي", "Pre-test")
    quiz("course.w09.pretest", [
        Q("قبل إدخال صور 0–255 إلى CNN:", ["لا شيء", "قسمة على 255 وإضافة بُعد القناة", "one-hot"], 1, ""),
        Q("60 صورة وشبكة بـ 10 آلاف معلمة:", ["تعميم ممتاز", "فرط تخصيص محتمل → زيادة/تنظيم", "قصور"], 1, ""),
        Q("الزيادة تُطبَّق على…", ["التدريب فقط", "الاختبار", "الكل"], 0, "التقييم على صور حقيقية."),
        Q("`expected min_ndim=4, found ndim=3`:", ["الدفعة كبيرة", "غابت قناة أو دفعة", "الخسارة خاطئة"], 1, ""),
    ], title_ar="الاختبار القبلي — الأسبوع 09")
    takeaway("الأسبوع 09 = مشروع صور كامل: تجهيز، بناء، تدريب، علاج فرط التخصيص، تفسير، وتصحيح.")
    lesson_footer(LESSON, ["المشروع في سبع محطات (تحريك).", "خريطة الأسبوع.", "الاختبار القبلي."])
