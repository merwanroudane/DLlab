from core.models import Module

MODULE = Module(
    id="foundations.training_loop",
    section="foundations",
    title_ar="الوحدة 16 — حلقة التدريب",
    title_en="Module 16 — The Training Loop",
    order=16,
    icon=":material/loop:",
    prerequisites=["foundations.backprop", "foundations.optim"],
    purpose_ar="الخطوات الاثنتا عشرة لحلقة التدريب: تهيئة ← دفعة ← أمامي ← خسارة ← خلفي ← محسّن ← تحديث ← دفعة تالية ← نهاية الحقبة ← تحقق ← حقبة تالية ← توقف؛ ثم كتابتها كاملة.",
    why_ar="`model.fit()` تخفي هذه الحلقة و`PyTorch` تجعلك تكتبها. من يعرفها يقرأ أي إطار.",
    objectives_ar=["ترتيب خطوات الحلقة وشرح كل واحدة.", "دور التحقق داخل الحلقة والإيقاف المبكر.", "كتابة حلقة تدريب كاملة بـ NumPy بتاريخ (history) كما في Keras."],
    challenges_ar=["وضع التحقق داخل حلقة الدفعات.", "تحديث المعلمات قبل حساب التدرج.", "نسيان الخلط كل حقبة."],
)

LESSON_MODULES = ["the_loop", "validation_in_loop", "training_loop_code"]
