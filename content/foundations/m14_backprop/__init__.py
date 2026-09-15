from core.models import Module

MODULE = Module(
    id="foundations.backprop",
    section="foundations",
    title_ar="الوحدة 14 — الانتشار الخلفي",
    title_en="Module 14 — Backpropagation",
    order=14,
    icon=":material/undo:",
    prerequisites=["foundations.calculus", "foundations.forward", "foundations.loss"],
    purpose_ar="حساب تدرج الخسارة بالنسبة لكل معلمة في الشبكة: قاعدة السلسلة على الرسم الحسابي، طبقةً طبقة إلى الخلف، مع مثال بخلية واحدة ثم بطبقتين، تحريك، تنفيذ كامل، ومشكلتا التلاشي والانفجار.",
    why_ar="`loss.backward()` و`GradientTape` يخفيان هذه الوحدة. من دونها تبقى «التدرجات تتلاشى» و«قصّ التدرج» تعليمات سحرية.",
    objectives_ar=["اشتقاق قواعد الانتشار الخلفي لطبقة كثيفة: δ، Xᵀδ، δWᵀ.", "تنفيذ backward كامل والتحقق منه عدديًا.", "تشخيص تلاشي/انفجار التدرج من معايير التدرج لكل طبقة."],
    challenges_ar=["الخلط بين اتجاه الأمامي والخلفي.", "نسيان ضرب مشتقة التنشيط.", "الاعتقاد أن الانتشار الخلفي يحدّث المعلمات (لا: يحسب التدرجات فقط؛ المحسّن يحدّث)."],
)

LESSON_MODULES = ["backpropagation", "bp_gradient_flow", "bp_one_neuron", "bp_two_layer", "bp_implementation", "bp_vanishing_exploding", "bp_debugging", "bp_practice"]
