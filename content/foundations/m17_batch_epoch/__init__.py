from core.models import Module

MODULE = Module(
    id="foundations.batch_epoch",
    section="foundations",
    title_ar="الوحدة 17 — الدفعة والحقبة والتكرار",
    title_en="Module 17 — Batch, Epoch & Iteration",
    order=17,
    icon=":material/view_module:",
    prerequisites=["foundations.training_loop"],
    purpose_ar="تفصيل كامل للمفاهيم الخمسة (حقبة، دفعة، حجم الدفعة، تكرار/خطوة، تحديث) بالمحاكاة، ثم أثر حجم الدفعة على الضوضاء والذاكرة والتعميم والحالات الحدّية.",
    why_ar="أكثر الأسئلة تكرارًا من الباحثين: «كم تحديثًا يحدث في الحقبة؟» و«ماذا أختار batch_size؟». هذه الوحدة تجيب بالأرقام.",
    objectives_ar=["حساب steps_per_epoch وعدد التحديثات الكلي لأي n وbatch_size وepochs.", "شرح الدفعة الجزئية وdrop_remainder وbatch_size=1 وbatch_size=n.", "اختيار حجم دفعة معقول وفهم علاقته بمعدل التعلم والذاكرة."],
    challenges_ar=["حقبة = تحديث.", "الدفعة الأكبر أفضل دائمًا.", "نسيان أن الحقبة الأخيرة قد تحتوي دفعة أصغر."],
)

LESSON_MODULES = ["definitions", "batch_size_effects"]
