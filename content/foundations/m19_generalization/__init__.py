from core.models import Module

MODULE = Module(
    id="foundations.generalization",
    section="foundations",
    title_ar="الوحدة 19 — التعميم",
    title_en="Module 19 — Generalization",
    order=19,
    icon=":material/balance:",
    prerequisites=["foundations.eval", "foundations.training_loop"],
    purpose_ar="منحنيات التدريب مقابل التحقق، قصور التعلم وفرط التخصيص، الانحياز والتباين، تعقيد النموذج وحجم البيانات، ومعمل تشخيص المنحنيات.",
    why_ar="الهدف الحقيقي ليس خسارة تدريب منخفضة بل أداء على بيانات جديدة. قراءة المنحنيات هي المهارة التشخيصية الأولى.",
    objectives_ar=["قراءة منحنيات التدريب/التحقق وتصنيفها: صحي، فرط تخصيص، قصور تعلم، معدل تعلم سيئ، تباعد، ضوضاء، هضبة.", "شرح الانحياز والتباين وربطهما بالقدرة والبيانات.", "تجربة أثر التعقيد وحجم البيانات على الفجوة."],
    challenges_ar=["الخلط بين الضوضاء الطبيعية للتحقق وفرط التخصيص.", "«زيادة البيانات دائمًا تحل المشكلة».", "قراءة منحنى واحد بلا الثاني."],
)

LESSON_MODULES = ["curves", "under_overfitting", "bias_variance", "complexity_data_size"]
