from components.quiz import Q
from components.week import week_overview
from core.models import Lesson

LESSON = Lesson(
    id="course.w12.overview",
    title_ar="نظرة عامة على الأسبوع 12 والاختبار القبلي",
    title_en="Week 12 Overview & Pre-test",
    module="course.w12",
    order=1,
    prerequisites=["course.w11.train_eval_diagnose"],
    objectives_ar=["خريطة الأسبوع.", "اختبار قبلي على GRU والمقارنة."],
    terms=["sequence"],
    difficulty="beginner",
    summary_ar="بوابتا التحديث وإعادة الضبط والحالة المخفية ← بناء وتدريب وتقييم ← مقارنة RNN/LSTM/GRU ← خريطة التطبيقات.",
)


def render() -> None:
    week_overview(
        LESSON, week_no="12",
        steps=["Update gate z", "Reset gate r", "Hidden state mix", "Build · train · evaluate", "RNN vs LSTM vs GRU", "Applications map", "Post-test"],
        links=[("بوابة التحديث وإعادة الضبط والحالة", "بوابتا GRU", "الأسبوع 11: البوابات؛ معمل بوابات GRU"), ("بناء وتدريب وتقييم ومقارنة", "المقارنة الثلاثية", "الأسبوعان 10–11: خط الأنابيب"), ("التطبيقات (أسعار، طقس، طاقة، لغة، ترجمة، توليد، كلام)", "التطبيقات", "الأسس 1: أنماط البيانات؛ الأسبوع 11: الإعداد")],
        buttons=[("معمل بوابات GRU", ":material/science:", "labs.gru_gates_lab"), ("الأسبوع 11 — البوابات", ":material/menu_book:", "course.w11.cell_gates_equations")],
        pretest=[Q("GRU لها…", ["ثلاث بوابات وحالتان", "بوابتان وحالة واحدة", "بوابة واحدة"], 1, ""), Q("h_t في GRU =", ["z⊙h̃ فقط", "(1−z)⊙h_{t−1} + z⊙h̃", "tanh(h_{t−1})"], 1, ""),
                 Q("GRU مقابل LSTM في المعلمات:", ["أكثر", "أقل (≈ 3/4)", "نفسها"], 1, ""), Q("اختيار بين LSTM وGRU يكون…", ["بالشهرة", "بالتجربة على التحقق", "GRU دائمًا"], 1, "")],
        note_ar="بعد الأسبوع 11 ستجد GRU مألوفة: نفس الأفكار بعدد أقل من القطع.",
        takeaway_ar="الأسبوع 12 = GRU كتبسيط لـ LSTM، ومقارنة ثلاثية تختم التسلسلات بقرار مبني على أدلة.")
