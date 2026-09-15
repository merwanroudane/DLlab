from components.quiz import Q
from components.week import week_overview
from core.models import Lesson

LESSON = Lesson(
    id="course.w11.overview",
    title_ar="نظرة عامة على الأسبوع 11 والاختبار القبلي",
    title_en="Week 11 Overview & Pre-test",
    module="course.w11",
    order=1,
    prerequisites=["course.w10.time_series_application"],
    objectives_ar=["خريطة الأسبوع.", "اختبار قبلي على البوابات وحالة الخلية."],
    terms=["sequence", "gradient"],
    difficulty="beginner",
    summary_ar="إعداد البيانات (نوافذ، حشو) ← حالة الخلية والبوابات الثلاث والمعادلات ← تحريك بوابةً بوابة ← بناء وتدريب وتقييم وتشخيص.",
)


def render() -> None:
    week_overview(
        LESSON, week_no="11",
        steps=["Data prep · windows · padding", "Cell state", "Forget · input · output gates", "Equations & gate animation", "Build & train", "Evaluate & diagnose", "Post-test"],
        links=[("إعداد البيانات: نوافذ التسلسل والحشو", "إعداد البيانات", "الأسبوع 10: النوافذ؛ الأسس 8"), ("حالة الخلية، البوابات، المعادلات، التحريك", "الخلية والبوابات", "الأسبوع 10: BPTT والتلاشي؛ الأسس 11: sigmoid/tanh"), ("بناء النموذج والتدريب والتقييم والتشخيص", "التدريب والتقييم", "الأسبوع 10: التطبيق؛ الأسبوع 07: المنهج")],
        buttons=[("معمل بوابات LSTM", ":material/science:", "labs.lstm_gates_lab"), ("الأسبوع 10 — BPTT", ":material/menu_book:", "course.w10.rnn_bptt_vanishing")],
        pretest=[Q("بوابة النسيان تُخرج قيمًا في…", ["(−1, 1)", "(0, 1)", "أي مدى"], 1, "sigmoid"), Q("حالة الخلية c_t تتحدث بـ…", ["ضرب متكرر في Wh", "جمع: f⊙c + i⊙g", "tanh فقط"], 1, ""),
                 Q("LSTM(32) على features=1: المعلمات مقابل SimpleRNN(32)", ["نفسها", "×4", "×2"], 1, ""), Q("تسلسلات نصية بأطوال مختلفة تحتاج…", ["حذف الطويلة", "حشوًا إلى طول موحّد (+ قناع)", "لا شيء"], 1, "")],
        note_ar="المعمل يجعل البوابات مرئية بأرقام حقيقية؛ لا تحفظ المعادلات قبل أن تراها تعمل.",
        takeaway_ar="الأسبوع 11 = ذاكرة ببوابات: تنسى وتكتب وتكشف بقرار متعلَّم.")
