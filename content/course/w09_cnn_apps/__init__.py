from core.models import Module

MODULE = Module(
    id="course.w09",
    section="course",
    title_ar="الأسبوع 09 — تطبيقات CNN",
    title_en="Week 09 — CNN Applications",
    order=9,
    icon=":material/image:",
    prerequisites=["course.w08", "foundations.regularization"],
    purpose_ar="مشروع صور مصغّر كامل: معالجة الصور مسبقًا، بناء CNN في Keras، التدريب، التقييم، فرط التخصيص، زيادة البيانات، خرائط الخصائص، وتصحيح أخطاء الأشكال — على صور اصطناعية صغيرة تُدرَّب في ثوانٍ، مع الإشارة إلى صور/رسوم التخصص.",
    why_ar="الأسبوع 08 مفاهيم؛ هنا CNN حقيقية تُدرَّب وتُقيَّم وتُشخَّص وتُفسَّر — بنفس منهج الأسبوع 07 لكن على موترات رتبة 4.",
    objectives_ar=["تجهيز صور كموترات (تحجيم 0–1، بُعد القناة، الدفعة) وبناء CNN وتدريبها.", "تشخيص فرط التخصيص على بيانات قليلة وعلاجه بزيادة البيانات والتنظيم.", "قراءة خرائط الخصائص والنوى المتعلَّمة، وإصلاح أخطاء الأشكال الشائعة في CNN."],
    challenges_ar=["نسيان بُعد القناة.", "زيادة بيانات لا تناسب المسألة (قلب صورة رقم 6)."],
)

LESSON_MODULES = ["overview", "build_train", "overfit_augment", "feature_maps_debugging"]
