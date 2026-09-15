from core.models import Module

MODULE = Module(
    id="course.w03",
    section="course",
    title_ar="الأسبوع 03 — Keras وTensorFlow مع تمهيد PyTorch المقارن",
    title_en="Week 03 — Keras & TensorFlow with a Comparative PyTorch Orientation",
    order=3,
    icon=":material/looks_3:",
    prerequisites=["course.w02", "foundations.frameworks"],
    purpose_ar="Keras كواجهة عليا وTensorFlow كمنظومة موترات/اشتقاق/بيانات/أجهزة؛ أول شبكة عصبية كاملة بتفكيك كل سطر وكل معامل ومخرج (summary، compile، fit، History، evaluate، predict، callbacks)؛ GradientTape وtf.data والأجهزة كتفسير لما تخفيه fit؛ ونفس الشبكة الأولى بـ PyTorch جنبًا إلى جنب.",
    why_ar="هذا الأسبوع يحوّل الوحدة 21 من معرفة إلى عادة عمل: كل مشروع لاحق في المقرر يبدأ بهذا القالب.",
    objectives_ar=["بناء أول شبكة في Keras وتفسير كل مخرج.", "شرح ما يحدث تحت fit بـ GradientTape وtf.data.", "كتابة المكافئ في PyTorch وتشخيص الأخطاء الشائعة الخمسة."],
    challenges_ar=["نسخ القالب دون الإجابة عن أسئلة «قبل الكود».", "الخلط بين Keras وTensorFlow وبيئة التشغيل."],
)

LESSON_MODULES = ["overview", "first_network", "under_the_hood", "pytorch_equivalent"]
