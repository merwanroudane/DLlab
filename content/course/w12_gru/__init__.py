from core.models import Module

MODULE = Module(
    id="course.w12",
    section="course",
    title_ar="الأسبوع 12 — GRU",
    title_en="Week 12 — GRU",
    order=12,
    icon=":material/sync_alt:",
    prerequisites=["course.w11"],
    purpose_ar="تحليل البيانات الزمنية أو النصية بـ GRU: إعداد البيانات، بناء النموذج، بوابة التحديث، بوابة إعادة الضبط، الحالة المخفية، التدريب، التقييم، ومقارنة RNN وLSTM وGRU — مع خريطة تطبيقات (أسعار، طقس، طاقة، لغة، ترجمة، توليد نص، كلام).",
    why_ar="GRU تبسّط LSTM إلى بوابتين وحالة واحدة بأداء مماثل غالبًا ومعلمات أقل. المقارنة الثلاثية تختم وحدة التسلسلات بقرار مبني على أدلة.",
    objectives_ar=["معادلات GRU الأربع ومعنى بوابتي التحديث وإعادة الضبط.", "مقارنة RNN/LSTM/GRU على نفس البيانات بالمعلمات والزمن والخطأ.", "تحديد نوع النموذج والمخرج لتطبيقات التسلسل الشائعة."],
    challenges_ar=["الخلط بين z في GRU وi/f في LSTM.", "اختيار البنية بالشهرة لا بالتحقق."],
)

LESSON_MODULES = ["overview", "update_reset_gates", "compare_rnn_lstm_gru", "applications"]
