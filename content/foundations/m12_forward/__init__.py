from core.models import Module

MODULE = Module(
    id="foundations.forward",
    section="foundations",
    title_ar="الوحدة 12 — التمرير الأمامي",
    title_en="Module 12 — Forward Pass",
    order=12,
    icon=":material/arrow_forward:",
    prerequisites=["foundations.architecture", "foundations.activations"],
    purpose_ar="كيف تتحول الدفعة إلى تنبؤ: مدخل ← مجموع موزون ← انحياز ← تنشيط ← الطبقة التالية ← logits ← احتمال ← عتبة ← فئة، بمثال رقمي كامل وتحريك وكود.",
    why_ar="الانتشار الخلفي والتشخيص كلاهما يفترض أنك تعرف بالضبط ما حدث في الأمامي وأشكاله عند كل خطوة.",
    objectives_ar=["تنفيذ تمرير أمامي كامل يدويًا بالأرقام ثم بالكود.", "التمييز بين logits واحتمال وفئة متنبأة.", "تتبع الأشكال عند كل طبقة لدفعة."],
    challenges_ar=["الخلط بين z وa.", "تفسير logits كاحتمالات.", "إسقاط محور الدفعة."],
)

LESSON_MODULES = ["layer_by_layer", "logits_probability_threshold", "numerical_walkthrough", "forward_pass_code"]
