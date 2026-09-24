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
    id="course.w10.overview",
    title_ar="نظرة عامة على الأسبوع 10 والاختبار القبلي",
    title_en="Week 10 Overview & Pre-test",
    module="course.w10",
    order=1,
    prerequisites=["course.w09.shape_debugging", "foundations.data.data_modalities"],
    objectives_ar=["رؤية أين تقف RNN بين MLP وCNN: مشاركة الأوزان عبر الزمن بدل المكان (تحريك).", "خريطة الأسبوع وروابط الأسس.", "اختبار قبلي على التسلسلات والحالة المخفية."],
    terms=["sequence", "timestep", "hidden_state", "rnn", "time_series"],
    difficulty="intermediate",
    summary_ar="التسلسل والنوافذ ← الحالة المخفية والخلية المنشورة ← BPTT وتلاشي التدرج ← سلسلة زمنية حقيقية مقابل الخط الساذج.",
)

_FAM = [
    ("MLP", "**MLP**: كل خاصية بوزن مستقل. لا يعرف «جوارًا» ولا «ترتيبًا».", K.BLUE),
    ("CNN", "**CNN**: نفس النواة تتكرر عبر **المكان** — لأن النمط قد يظهر في أي موضع من الصورة.", K.VIOLET),
    ("RNN", "**RNN**: نفس الخلية تتكرر عبر **الزمن** — لأن النمط قد يظهر في أي لحظة من التسلسل، والماضي يؤثر في الحاضر.", K.PINK),
    ("memory", "**الذاكرة**: الحالة المخفية h تنقل ملخص الماضي من خطوة إلى التالية.", K.AMBER),
    ("limit", "**الحد**: ذاكرة RNN البسيطة قصيرة — التدرج يتلاشى عبر الزمن. LSTM وGRU (الأسبوعان 11–12) هما الحل.", K.RED),
]


def _svg(k: int) -> str:
    s = K.svg_open(700, 130)
    for i, (n, _, col) in enumerate(_FAM):
        x = 10 + i * 138
        s += K.box(x, 40, 124, 46, n, col, filled=i == k, size=13)
        if i < len(_FAM) - 1:
            s += K.arrow(x + 125, 63, x + 137, 63, color=col)
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["Sequences & windows", "Hidden state & recurrence", "Unrolling & BPTT", "Vanishing / exploding", "RNN on a real series", "Post-test"], active=0)
    h2("مشاركة الأوزان: من المكان إلى الزمن", "Weight sharing: from space to time")
    animation_player("w10_family", [Frame(_svg(i), caption(t), action=n) for i, (n, t, _) in enumerate(_FAM)], title_ar="أين تقف RNN؟", interval_ms=2300)
    table(["الموضوع", "درس الأسبوع", "الأسس التي يعتمد عليها"],
          [("التسلسل والنافذة والشكل (samples, timesteps, features)", "التسلسل والحالة المخفية", "الأسس 1: السلاسل الزمنية؛ الأسس 4: الموترات"), ("الخلية والحالة المخفية والاتصال التكراري", "التسلسل والحالة المخفية", "الأسس 9 و12: الخلية والتمرير الأمامي"),
           ("BPTT والتلاشي والانفجار", "BPTT والتلاشي", "الأسس 14: الانتشار الخلفي؛ الأسبوع 06: التلاشي"), ("تطبيق على سلسلة زمنية", "التطبيق", "الأسبوع 02: خط الأساس والتقسيم")],
          ["rtl", "rtl", "rtl"])
    st.markdown(
        "**في نهاية الأسبوع ستستطيع:**\n"
        "- أن تحوّل سلسلة إلى نوافذ بشكل صحيح وتقسمها زمنيًا بلا تسريب.\n"
        "- أن تحسب خطوة RNN يدويًا وتشرح دور الحالة المخفية.\n"
        "- أن تشرح لماذا تتلاشى التدرجات عبر الزمن وما العلاجات.\n"
        "- أن تقيّم RNN بصدق مقابل الخط الساذج."
    )
    practical_note("السلاسل الاقتصادية قصيرة وضوضائية غالبًا: هذا الأسبوع يعلّمك أيضًا متى **لا** تستعمل RNN — وهي مهارة بنفس الأهمية.")
    with st.container(horizontal=True):
        st.button("معمل نشر RNN", icon=":material/science:", on_click=go, args=("labs.rnn_unrolling_lab",), key="w10_go_lab")
    intuition("RNN = حلقة for على الزمن داخلها خلية صغيرة. كل ما تعرفه عن الطبقات والتدرجات ينطبق؛ الجديد فقط أن نفس الأوزان تُستعمل في كل دورة.")
    h2("الاختبار القبلي", "Pre-test")
    quiz("course.w10.pretest", [
        Q("شكل دفعة تسلسلات في Keras:", ["(samples, features)", "(samples, timesteps, features)", "(timesteps, samples)"], 1, "رتبة 3."),
        Q("h_t تعتمد على…", ["x_t فقط", "x_t وh_{t−1}", "y_t"], 1, "المدخل الحالي والذاكرة."),
        Q("أوزان RNN عبر الخطوات…", ["مختلفة", "نفسها", "عشوائية"], 1, "مشاركة عبر الزمن."),
        Q("تقسيم سلسلة زمنية:", ["عشوائي", "زمني", "بالتساوي"], 1, ""),
        Q("180 شهرًا ونافذة 12: عدد النوافذ", ["180", "168", "15"], 1, "180 − 12."),
    ], title_ar="الاختبار القبلي — الأسبوع 10")
    takeaway("الأسبوع 10 = التسلسل كنافذة، الخلية كحلقة بذاكرة، التدرج عبر الزمن، وتقييم صادق مقابل الساذج.")
    lesson_footer(LESSON, ["مشاركة الأوزان عبر الزمن (تحريك).", "خريطة الأسبوع.", "الاختبار القبلي."])
