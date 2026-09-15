from components.quiz import Q
from components.week import week_overview
from core.models import Lesson

LESSON = Lesson(
    id="course.w06.overview",
    title_ar="نظرة عامة على الأسبوع 06 والاختبار القبلي",
    title_en="Week 06 Overview & Pre-test",
    module="course.w06",
    order=1,
    prerequisites=["course.w05.loss_surface_convergence", "foundations.activations.why_nonlinearity"],
    objectives_ar=["خريطة الأسبوع وروابط الأسس.", "اختبار قبلي على دوال التنشيط."],
    terms=["softmax", "derivative"],
    difficulty="beginner",
    summary_ar="Sigmoid/Tanh/Softmax ← ReLU وعائلته ← Swish/ELU/SELU ← المشتقات والتشبع وReLU الميت ← الاختيار حسب المهمة.",
)


def render() -> None:
    week_overview(
        LESSON, week_no="06",
        steps=["Sigmoid · Tanh · Softmax", "ReLU · LeakyReLU", "Swish · ELU · SELU", "Derivatives & saturation", "Dead ReLU", "Choosing by task", "Post-test"],
        links=[("Sigmoid وTanh وSoftmax", "الدوال ومشتقاتها", "الأسس 11: sigmoid/tanh، softmax"), ("ReLU وLeaky ReLU وELU وSELU وSwish", "الدوال ومشتقاتها", "الأسس 11: عائلة ReLU، Swish/GELU"),
               ("المشتقات والتشبع وReLU الميت", "الدوال ومشتقاتها", "الأسس 11: المشتقات والتشبع؛ الأسس 14: التلاشي"), ("الاختيار حسب المهمة", "اختيار التنشيط", "الأسس 11 و13: توافق المخرج والخسارة")],
        buttons=[("الأسس 11 — التنشيط", ":material/menu_book:", "foundations.activations"), ("معمل التنشيط", ":material/science:", "labs.activation_lab")],
        pretest=[Q("مشتقة sigmoid القصوى:", ["1", "0.25", "0.5"], 1, ""), Q("ReLU الميت:", ["وحدة مخرجها دائمًا 0 وتدرجها 0", "وحدة بطيئة", "وحدة بلا انحياز"], 0, ""),
                 Q("تنشيط المخرج لتصنيف 5 فئات متنافية:", ["sigmoid", "softmax", "ReLU"], 1, ""), Q("Tanh مقابل sigmoid: المخرج", ["(0,1)", "(−1,1)", "(0,∞)"], 1, "")],
        note_ar="الأسس 11 تشرح كل دالة؛ هذا الأسبوع يقيس أثرها على شبكة عميقة حقيقية ويعطيك قاعدة الاختيار.",
        takeaway_ar="الأسبوع 06 = الدوال بمشتقاتها، أثرها على العمق، وقاعدة اختيار بحسب المهمة.")
