import streamlit as st

from components import svgkit as K
from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, practical_note, takeaway, why
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline, table

LESSON = Lesson(
    id="course.w08.overview",
    title_ar="نظرة عامة على الأسبوع 08 والاختبار القبلي",
    title_en="Week 08 Overview & Pre-test",
    module="course.w08",
    order=1,
    prerequisites=["course.w07.diagnose_interpret", "foundations.linalg.tensors", "foundations.data.shape_axis_rank"],
    objectives_ar=["رؤية لماذا يفشل MLP على الصور بمعلماته وحساسيته للإزاحة، وما يعدّه CNN (تحريك).", "خريطة الأسبوع وروابط الأسس.", "اختبار قبلي على الموترات والالتفاف."],
    terms=["tensor", "channel_dimension", "shape", "cnn", "convolution"],
    difficulty="beginner",
    summary_ar="الصورة كموتر ← النواة والالتفاف وخريطة الخصائص ← الحشو والخطوة والتجميع ← الطبقات والبنى وحساب الأشكال.",
)

_WHY = [
    ("pixels", "**الصورة شبكة بكسلات**: صورة 28×28 = 784 رقمًا. صورة هاتف 1000×1000×3 = 3 ملايين رقم.", K.BLUE),
    ("MLP", "**MLP يسطّحها**: كل وحدة مخفية ترى كل البكسلات. وحدة واحدة على صورة هاتف = 3 ملايين وزن؛ طبقة من 100 وحدة = 300 مليون.", K.RED),
    ("shift", "**إزاحة بكسل واحد** تجعل المدخل «جديدًا» كليًا لـ MLP: البكسل (5، 7) والبكسل (5، 8) لا علاقة بينهما عنده.", K.ORANGE),
    ("local", "**CNN: محلية** — كل كاشف يرى نافذة 3×3 صغيرة فقط، لأن البكسلات المتجاورة هي المترابطة.", K.EMERALD),
    ("shared", "**CNN: مشاركة الأوزان** — نفس الكاشف (9 أوزان) يمسح كل المواضع: حافة في الزاوية وحافة في الوسط تُكشفان بنفس النواة.", K.VIOLET),
    ("hierarchy", "**CNN: هرمية** — حواف ← زوايا وأشكال ← أجزاء ← كائنات، طبقة بعد طبقة، مع تقليص الحجم.", K.PINK),
]


def _svg(k: int) -> str:
    s = K.svg_open(700, 150)
    for i, (n, _, col) in enumerate(_WHY):
        x = 8 + i * 115
        s += K.box(x, 40, 104, 50, n, col, filled=i == k, size=12)
        if i < len(_WHY) - 1:
            s += K.arrow(x + 105, 65, x + 114, 65, color=col)
    s += K.text(175, 125, "the problem", size=12, bold=True, color=K.RED)
    s += K.text(520, 125, "the CNN answer", size=12, bold=True, color=K.EMERALD)
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["Image as tensor", "Kernel & convolution", "Feature maps", "Padding · stride · pooling", "CNN layers & architectures", "Shape calculation", "Post-test"], active=0)
    h2("لماذا لا يكفي MLP للصور؟", "Why an MLP is not enough for images")
    animation_player("w08_why", [Frame(_svg(i), caption(t), action=n, highlight=None) for i, (n, t, _) in enumerate(_WHY)], title_ar="المشكلة وجواب CNN", interval_ms=2400)
    why("الالتفاف ليس «شبكة من نوع آخر»: هو طبقة كثيفة بقيدين — كل وحدة ترى جوارًا صغيرًا فقط، والوحدات تتشارك أوزانها. هذان القيدان يطابقان طبيعة الصور فيقللان المعلمات بآلاف المرات ويجعلان التعلم ممكنًا ببيانات معقولة.")
    table(["الموضوع", "درس الأسبوع", "الأسس التي يعتمد عليها"],
          [("الصورة كموتر، البكسل، القناة", "الالتفاف", "الأسس 1 و4: الشكل والمحاور، بُعد القناة"), ("النواة/المرشّح، الالتفاف، خريطة الخصائص، تحريك النواة", "الالتفاف", "الأسس 4: الضرب العنصري والجمع؛ الأسس 9: المجموع الموزون"),
           ("الحشو، الخطوة، التجميع", "الحشو والخطوة والتجميع", "معمل الحشو والخطوة"), ("الطبقات الأساسية والبنى النموذجية وحساب الأشكال", "الطبقات والأشكال", "الأسس 10: عدّ المعلمات؛ الوحدة 21: summary")],
          ["rtl", "rtl", "rtl"])
    st.markdown(
        "**في نهاية الأسبوع ستستطيع:**\n"
        "- أن تحسب خلية في خريطة خصائص يدويًا، لقناة واحدة ولعدة قنوات.\n"
        "- أن تتنبأ بحجم أي خريطة بصيغة واحدة بعد الحشو والخطوة والتجميع.\n"
        "- أن تتتبع الأشكال والمعلمات في CNN كاملة وتطابقها مع summary.\n"
        "- أن تشرح مجال الرؤية ولماذا تتفوق CNN على MLP في الصور."
    )
    practical_note("أول بنية غير MLP: كل ما تحتاجه هو الجبر الخطي (الضرب العنصري والجمع) وحساب الأشكال. المعامل الثلاثة تجعل النواة تتحرك أمام عينيك.")
    with st.container(horizontal=True):
        st.button("معمل الالتفاف (النواة المتحركة)", icon=":material/science:", on_click=go, args=("labs.cnn_convolution_lab",), key="w08_go_conv")
        st.button("حاسبة أشكال CNN", icon=":material/science:", on_click=go, args=("labs.cnn_shape_calculator",), key="w08_go_calc")
        st.button("الأسس 4 — الموترات", icon=":material/menu_book:", on_click=go, args=("foundations.linalg.tensors",), key="w08_go_tens")
    intuition("كل CNN تُقرأ بسؤالين: ماذا يكشف كل مرشّح (المحتوى)، وكيف تتغير الأشكال (الهندسة). هذا الأسبوع للهندسة والآلية؛ الأسبوع 09 لما تتعلمه الشبكة فعلًا.")
    h2("الاختبار القبلي", "Pre-test")
    quiz("course.w08.pretest", [
        Q("صورة ملونة 32×32 كموتر (Keras):", ["(32, 32)", "(32, 32, 3)", "(3, 32)"], 1, "channels-last."),
        Q("نواة 3×3 على قناة واحدة: معلماتها", ["9", "10", "3"], 1, "9 أوزان + انحياز."),
        Q("الالتفاف ينتج…", ["رقمًا واحدًا", "خريطة خصائص", "متجه أوزان"], 1, ""),
        Q("MaxPooling(2) على 28×28 يعطي", ["28×28", "14×14", "26×26"], 1, ""),
        Q("لماذا لا نستعمل MLP على صور كبيرة؟", ["لا يعمل إطلاقًا", "معلمات هائلة وحساسية للإزاحة", "أبطأ فقط"], 1, ""),
    ], title_ar="الاختبار القبلي — الأسبوع 08")
    takeaway("الأسبوع 08 = الصورة موتر، النواة كاشف نمط متحرك، والأشكال تُحسب بصيغة واحدة.")
    lesson_footer(LESSON, ["لماذا CNN (تحريك).", "خريطة الأسبوع.", "الاختبار القبلي."])
