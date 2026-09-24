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
    id="course.w05.overview",
    title_ar="نظرة عامة على الأسبوع 05 والاختبار القبلي",
    title_en="Week 05 Overview & Pre-test",
    module="course.w05",
    order=1,
    prerequisites=["course.w04.depth_width", "foundations.optim.learning_rate", "foundations.optim.rmsprop_adam"],
    objectives_ar=["رؤية أسئلة التحسين الأربعة: أين نحن؟ إلى أين؟ كم خطوة؟ بأي ذاكرة؟ (تحريك).", "خريطة الأسبوع وروابط الأسس.", "اختبار قبلي على الانحدار التدرجي والمحسّنات."],
    terms=["gradient", "learning_rate", "optimizer", "loss"],
    difficulty="intermediate",
    summary_ar="الانحدار التدرجي وη ← SGD والدفعات ← الزخم وRMSprop وAdam ← سطح الخسارة والتقارب ← الجداول ← سباق المحسّنات.",
)

_Q = [
    ("where", "**أين نحن؟** قيمة الخسارة الحالية L(θ) على السطح.", K.VIOLET),
    ("direction", "**إلى أين؟** عكس التدرج −∇L: أشد انحدار محليًا (الانحدار التدرجي).", K.BLUE),
    ("step size", "**كم نخطو؟** معدل التعلم η: صغير = بطيء، كبير = تذبذب/انفجار.", K.AMBER),
    ("estimate", "**على كم ملاحظة نقدّر التدرج؟** كل البيانات (GD) أم دفعة (SGD): دقة مقابل سرعة وضوضاء.", K.CYAN),
    ("memory", "**هل نتذكر الماضي؟** الزخم يراكم الاتجاه، RMSprop يكيّف η لكل معلمة، Adam يجمعهما.", K.PINK),
    ("schedule", "**هل يتغير η مع الوقت؟** الجداول: كبير في البداية، صغير في النهاية.", K.EMERALD),
]


def _svg(k: int) -> str:
    s = K.svg_open(680, 150)
    for i, (n, _, col) in enumerate(_Q):
        x = 8 + i * 112
        s += K.box(x, 45, 100, 50, n, col, filled=i == k, size=12)
        if i < len(_Q) - 1:
            s += K.arrow(x + 101, 70, x + 111, 70, color=col)
    s += K.text(340, 130, "θ ← θ − η · (estimated, remembered) ∇L", size=13, mono=True, bold=True, color=K.INK)
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["Gradient descent & η", "SGD & batches", "Momentum · RMSprop · Adam", "Loss surface & convergence", "LR problems & schedules", "Optimizer race", "Post-test"], active=0)
    h2("ستة أسئلة يجيب عنها كل محسّن", "Six questions every optimizer answers")
    animation_player("w05_questions", [Frame(_svg(i), caption(t), action=n, highlight=i) for i, (n, t, _) in enumerate(_Q)],
                     title_ar="تشريح خطوة التحسين", stages=[q[0] for q in _Q], interval_ms=2300)
    table(["الموضوع", "درس الأسبوع", "الأسس التي يعتمد عليها"],
          [("الانحدار التدرجي ومعدل التعلم وشرط التقارب", "GD وη وSGD", "الأسس 15: معدل التعلم، GD وأنواعه"), ("SGD والدفعات والضوضاء", "GD وη وSGD", "الأسس 17: الدفعة والحقبة"), ("الزخم وRMSprop وAdam", "الزخم وRMSprop وAdam", "الأسس 15: الزخم، RMSprop/Adam"),
           ("الخسارة وأثرها، سطح الخسارة، التقارب", "سطح الخسارة والتقارب", "الأسس 13، 15: سطح الخسارة، الجداول والتقارب"), ("سباق المحسّنات", "الزخم وRMSprop وAdam", "معمل سباق المحسّنات")],
          ["rtl", "rtl", "rtl"])
    st.markdown(
        "**في نهاية الأسبوع ستستطيع:**\n"
        "- أن تتنبأ بسلوك η على سطح بسيط وتشتق شرط التقارب.\n"
        "- أن تشرح ما يعالجه الزخم وRMSprop وAdam وتحسب خطوة Adam يدويًا.\n"
        "- أن تقرأ منحنى الخسارة كطبيب: تقارب، هضبة، تذبذب، انفجار — وتختار العلاج.\n"
        "- أن تختار جدولًا لمعدل التعلم وتضبطه في Keras."
    )
    practical_note("الأسس 15 تشرح كل محسّن رياضيًا وبالتحريك؛ هذا الأسبوع يشغّلها على شبكة Keras حقيقية ويقرأ منحنيات الخسارة كطبيب.")
    with st.container(horizontal=True):
        st.button("الأسس 15 — التحسين", icon=":material/menu_book:", on_click=go, args=("foundations.optim",), key="w05_go_opt")
        st.button("معمل سباق المحسّنات", icon=":material/science:", on_click=go, args=("labs.optimizer_race",), key="w05_go_race")
        st.button("معمل معدل التعلم", icon=":material/science:", on_click=go, args=("labs.learning_rate_lab",), key="w05_go_lr")
    intuition("كل محسّن هو نفس السطر θ ← θ − η·(شيء يشبه التدرج). الفرق كله في «الشيء»: تدرج خام، أو دفعة، أو ذاكرة، أو تكييف.")
    h2("الاختبار القبلي", "Pre-test")
    quiz("course.w05.pretest", [
        Q("خطوة الانحدار التدرجي:", ["θ ← θ + η∇L", "θ ← θ − η∇L", "θ ← ∇L"], 1, "عكس التدرج."),
        Q("η كبير جدًا يسبب…", ["تقاربًا بطيئًا", "تذبذبًا أو انفجارًا", "لا شيء"], 1, ""),
        Q("الزخم يساعد على…", ["تسريع الاتجاهات المتسقة وتخفيف التذبذب", "زيادة الضوضاء", "تقليل المعلمات"], 0, ""),
        Q("Adam يجمع بين…", ["الزخم وتكييف η لكل معلمة", "L2 وDropout", "batch وepoch"], 0, ""),
        Q("SGD بدفعة 32 مقابل كل البيانات:", ["أبطأ لكل تحديث", "تحديثات أكثر لكل حقبة مع ضوضاء", "نفس الشيء"], 1, ""),
    ], title_ar="الاختبار القبلي — الأسبوع 05")
    takeaway("الأسبوع 05 = السلوك المرئي لكل محسّن ومعدل تعلم، ثم تطبيقه في compile وقراءة منحنى الخسارة.")
    lesson_footer(LESSON, ["ستة أسئلة للمحسّن (تحريك).", "خريطة الأسبوع.", "الاختبار القبلي."])
