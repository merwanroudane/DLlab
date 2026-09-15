from core.models import Module

MODULE = Module(
    id="course.w06",
    section="course",
    title_ar="الأسبوع 06 — دوال التنشيط",
    title_en="Week 06 — Activation Functions",
    order=6,
    icon=":material/looks_6:",
    prerequisites=["course.w05", "foundations.activations"],
    purpose_ar="Sigmoid وSoftmax وTanh وReLU وLeaky ReLU وSwish وELU وSELU: المقارنة، المشتقات، التشبع، ReLU الميت، واختيار التنشيط المناسب حسب المهمة — مع تجربة حقيقية في Keras على شبكة عميقة.",
    why_ar="التنشيط يقرر إن كانت التدرجات تصل إلى الطبقات الأولى أم تتلاشى، وإن كان المخرج احتمالًا أم رقمًا. اختيار خاطئ في المخرج خطأ صامت؛ اختيار خاطئ في الوسط تدريب معطَّل.",
    objectives_ar=["رسم كل دالة ومشتقتها وتحديد مناطق التشبع.", "قياس أثر التنشيط المخفي على شبكة عميقة (تلاشي التدرج، ReLU الميت).", "قاعدة اختيار التنشيط للمخرج بحسب المهمة وللمخفي بحسب العمق."],
    challenges_ar=["sigmoid في الطبقات المخفية العميقة.", "softmax مع وحدة مخرج واحدة."],
)

LESSON_MODULES = ["overview", "functions_derivatives", "choosing_activations"]
