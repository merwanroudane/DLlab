from core.models import Module

MODULE = Module(
    id="course.w11",
    section="course",
    title_ar="الأسبوع 11 — LSTM",
    title_en="Week 11 — LSTM",
    order=11,
    icon=":material/memory:",
    prerequisites=["course.w10"],
    purpose_ar="تحليل البيانات الزمنية (والنصية) بـ LSTM: إعداد البيانات، نوافذ التسلسل، الحشو عند الحاجة، بناء النموذج، حالة الخلية، بوابات النسيان/الإدخال/الإخراج، المعادلات، تحريك بوابةً بوابة، التدريب، التقييم، والتشخيص.",
    why_ar="LSTM هي الجواب البنيوي على تلاشي التدرج: حالة خلية تتحدث بالجمع وبوابات تتعلم متى تنسى وتكتب وتكشف. من يفهم البوابات يقرأ أي بنية تكرارية حديثة.",
    objectives_ar=["كتابة معادلات LSTM الأربع وتفسير كل بوابة بقيم حقيقية.", "إعداد بيانات تسلسلية (نوافذ، حشو للأطوال المختلفة) وبناء LSTM في Keras.", "تدريب وتقييم وتشخيص LSTM على سلسلة زمنية ومقارنتها بـ RNN وبالساذج."],
    challenges_ar=["حفظ المعادلات دون معنى البوابات.", "توقع تفوق LSTM تلقائيًا على بيانات قصيرة."],
)

LESSON_MODULES = ["overview", "data_prep_windows_padding", "cell_gates_equations", "train_eval_diagnose"]
