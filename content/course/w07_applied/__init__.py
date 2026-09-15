from core.models import Module

MODULE = Module(
    id="course.w07",
    section="course",
    title_ar="الأسبوع 07 — تطبيق عملي عام",
    title_en="Week 07 — General Applied Project",
    order=7,
    icon=":material/looks_one:",
    prerequisites=["course.w06", "foundations.prep", "foundations.eval", "foundations.regularization"],
    purpose_ar="بناء نموذج تنبؤ (انحدار) ونموذج تصنيف على بيانات بحسب التخصص: إعداد البيانات، خط الأساس، البنية، التدريب، التقييم، التشخيص، التفسير، وتحليل الأخطاء — كقالب كامل يُعاد استخدامه في مشروع الأسبوع 14.",
    why_ar="الأسابيع 02–06 قطع؛ هذا الأسبوع يركّبها في مشروع مصغّر كامل بمنهج علمي: كل قرار مبرَّر، كل رقم على بيانات محجوزة، وكل خطأ مُحلَّل.",
    objectives_ar=["تنفيذ خط أنابيب انحدار كامل على أسعار العقارات بخط أساس ونموذج Keras وتقييم بوحدة الهدف.", "تشخيص النموذج من منحنياته وتحليل أخطائه وتفسير خصائصه بأهمية التبديل.", "كتابة تقرير نتائج قابل للاستنساخ."],
    challenges_ar=["الإبلاغ عن الخسارة بدل مقياس مفهوم بوحدة الهدف.", "القفز إلى التفسير السببي."],
)

LESSON_MODULES = ["overview", "build_pipeline", "diagnose_interpret"]
