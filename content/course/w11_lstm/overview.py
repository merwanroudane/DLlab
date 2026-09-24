import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, practical_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from content.course.w11_lstm._viz import memory_svg, memory_test
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline, table

LESSON = Lesson(
    id="course.w11.overview",
    title_ar="نظرة عامة على الأسبوع 11 والاختبار القبلي",
    title_en="Week 11 Overview & Pre-test",
    module="course.w11",
    order=1,
    prerequisites=["course.w10.time_series_application"],
    objectives_ar=["رؤية المشكلة التي تحلها LSTM: RNN تنسى، LSTM تتذكر (تحريك سريع).", "خريطة الأسبوع.", "اختبار قبلي على البوابات وحالة الخلية."],
    terms=["sequence", "gradient", "lstm", "gate"],
    difficulty="intermediate",
    summary_ar="إعداد البيانات (نوافذ، متعدد المتغيرات، حشو وقناع) ← حالة الخلية والبوابات الثلاث ← التدريب والتقييم والتشخيص مقابل RNN والساذج.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["Data prep · windows · padding", "Cell state", "Forget · input · output gates", "Equations & gate animation", "Build & train", "Evaluate & diagnose", "Post-test"], active=0)
    h2("المشكلة في عشرين خطوة", "The problem in twenty steps")
    mt = memory_test()
    shown = [0, 2, 5, 9, 14, 19]
    caps = [f"**t = {k + 1}**: ما بقي من نبضة البداية — RNN {mt['h_rnn'][k]:.2f}، LSTM {mt['c'][k]:.2f}." for k in shown]
    caps[0] += " الأسبوع 10 انتهى هنا: RNN البسيطة تنسى بسرعة والتدرج يتلاشى."
    caps[-1] += " هذا الأسبوع يشرح **كيف** تحمل LSTM الذاكرة: حالة خلية تتحدث بالجمع وبوابات تتعلم متى تنسى."
    animation_player("w11_teaser", [Frame(memory_svg(mt, k), caption(c), action=f"t = {k + 1}") for k, c in zip(shown, caps)], title_ar="من يتذكر نبضة واحدة؟", interval_ms=1600)
    table(["الموضوع", "درس الأسبوع", "الأسس التي يعتمد عليها"],
          [("إعداد البيانات: نوافذ التسلسل والحشو", "إعداد البيانات", "الأسبوع 10: النوافذ؛ الأسس 8"), ("حالة الخلية، البوابات، المعادلات، التحريك", "الخلية والبوابات", "الأسبوع 10: BPTT والتلاشي؛ الأسس 11: sigmoid/tanh"), ("بناء النموذج والتدريب والتقييم والتشخيص", "التدريب والتقييم", "الأسبوع 10: التطبيق؛ الأسبوع 07: المنهج")],
          ["rtl", "rtl", "rtl"])
    st.markdown(
        "**في نهاية الأسبوع ستستطيع:**\n"
        "- أن تعدّ نوافذ متعددة المتغيرات والخطوات، وتحشو وتقنّع تسلسلات بأطوال مختلفة.\n"
        "- أن تكتب معادلات LSTM وتشرح كل بوابة بقيم حقيقية.\n"
        "- أن تشرح لماذا يحفظ مسار الجمع التدرج.\n"
        "- أن تدرّب LSTM وتقارنها بصدق مع RNN والساذج."
    )
    practical_note("المعمل يجعل البوابات مرئية بأرقام حقيقية؛ لا تحفظ المعادلات قبل أن تراها تعمل.")
    with st.container(horizontal=True):
        st.button("معمل بوابات LSTM", icon=":material/science:", on_click=go, args=("labs.lstm_gates_lab",), key="w11_go_lab")
        st.button("الأسبوع 10 — BPTT", icon=":material/menu_book:", on_click=go, args=("course.w10.rnn_bptt_vanishing",), key="w11_go_bptt")
    intuition("LSTM = RNN + سير ناقل للذاكرة + ثلاثة صمامات. كل صمام مجموع موزون ثم sigmoid — لا شيء جديد رياضيًا، فقط تصميم ذكي.")
    h2("الاختبار القبلي", "Pre-test")
    quiz("course.w11.pretest", [
        Q("بوابة النسيان تُخرج قيمًا في…", ["(−1, 1)", "(0, 1)", "أي مدى"], 1, "sigmoid."),
        Q("حالة الخلية c_t تتحدث بـ…", ["ضرب متكرر في Wh", "جمع: f⊙c + i⊙g", "tanh فقط"], 1, ""),
        Q("LSTM(32) على features=1: المعلمات مقابل SimpleRNN(32)", ["نفسها", "×4", "×2"], 1, "أربع مجموعات أوزان."),
        Q("تسلسلات نصية بأطوال مختلفة تحتاج…", ["حذف الطويلة", "حشوًا إلى طول موحّد (+ قناع)", "لا شيء"], 1, ""),
    ], title_ar="الاختبار القبلي — الأسبوع 11")
    takeaway("الأسبوع 11 = ذاكرة ببوابات: تنسى وتكتب وتكشف بقرار متعلَّم.")
    lesson_footer(LESSON, ["المشكلة في عشرين خطوة (تحريك).", "خريطة الأسبوع.", "الاختبار القبلي."])
