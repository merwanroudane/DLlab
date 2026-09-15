from core.models import Module

MODULE = Module(
    id="foundations.activations",
    section="foundations",
    title_ar="الوحدة 11 — أسس دوال التنشيط",
    title_en="Module 11 — Activation Foundations",
    order=11,
    icon=":material/waves:",
    prerequisites=["foundations.neuron", "foundations.calculus"],
    purpose_ar="لماذا اللاخطية، ثم كل دالة تنشيط: صيغتها، مداها، مشتقتها، متى تُشبع أو تموت، وأيها يناسب طبقة الإخراج.",
    why_ar="اختيار التنشيط الخاطئ في الإخراج يجعل الخسارة بلا معنى؛ واختياره الخاطئ في المخفية يجعل التدرج يتلاشى أو يموت. هذه الوحدة تعطيك جدول القرار.",
    objectives_ar=["رسم وحساب Sigmoid وtanh وReLU وعائلته وSoftmax ومشتقاتها.", "تشخيص الإشباع وDead ReLU من المشتقات.", "اختيار تنشيط المخفية (ReLU افتراضيًا) وتنشيط الإخراج (حسب المهمة)."],
    challenges_ar=["Sigmoid في الطبقات المخفية العميقة.", "ReLU في الإخراج للانحدار بقيم سالبة.", "Softmax مع BCE."],
)

LESSON_MODULES = ["why_nonlinearity", "sigmoid_tanh", "relu_family", "swish_gelu", "softmax_output", "derivatives_saturation_dead"]
