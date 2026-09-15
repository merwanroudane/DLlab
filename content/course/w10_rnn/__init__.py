from core.models import Module

MODULE = Module(
    id="course.w10",
    section="course",
    title_ar="الأسبوع 10 — الشبكات العصبية التكرارية RNN",
    title_en="Week 10 — Recurrent Neural Networks",
    order=10,
    icon=":material/timeline:",
    prerequisites=["course.w09", "foundations.data", "foundations.backprop"],
    purpose_ar="التسلسل، الخطوة الزمنية، الترتيب، الحالة المخفية، الاتصال التكراري، النشر عبر الزمن، أساسيات RNN، تمهيد LSTM، التطبيق على البيانات الزمنية، BPTT، وتلاشي/انفجار التدرج.",
    why_ar="الجداول والصور لا ترتيب زمنيًا فيها؛ التضخم والمبيعات والنصوص تسلسلات يهم فيها الترتيب. RNN أول بنية تحمل «ذاكرة» — وفهم حدودها (التلاشي) هو مدخل LSTM وGRU.",
    objectives_ar=["تحويل سلسلة زمنية إلى نوافذ (samples, timesteps, features) وفهم الترتيب والخطوة والحالة.", "تنفيذ خلية RNN يدويًا ونشرها عبر الزمن، وفهم BPTT وتلاشي/انفجار التدرج.", "تدريب SimpleRNN في Keras على سلسلة زمنية ومقارنتها بخط الأساس الساذج."],
    challenges_ar=["خلط النوافذ الزمنية عشوائيًا قبل التقسيم (تسريب من المستقبل).", "توقع أن RNN يهزم «القيمة الأخيرة» تلقائيًا."],
)

LESSON_MODULES = ["overview", "sequences_hidden_state", "rnn_bptt_vanishing", "time_series_application"]
