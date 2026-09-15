from core.models import Module

MODULE = Module(
    id="course.w15",
    section="course",
    title_ar="الأسبوع 15 — عرض ومناقشة المشاريع",
    title_en="Week 15 — Project Presentation & Discussion",
    order=15,
    icon=":material/co_present:",
    prerequisites=["course.w14"],
    purpose_ar="عرض المشاريع، تقييم الأداء، تحليل النتائج، تشخيص نقاط الضعف، مقارنة الاختيارات، مقترحات التطوير، المناقشة، الرمز النهائي، وجودة الاستنساخ والعرض.",
    why_ar="النتيجة العلمية لا تكتمل حتى تُعرض وتُناقَش وتُعاد. هذا الأسبوع يحوّل المشروع إلى عرض مقنع ونقاش نزيه، ويعلن معايير التقييم كاملة.",
    objectives_ar=["بناء عرض من 10 شرائح يتبع المنهج ويُظهر الأدلة.", "المناقشة: الأسئلة المتوقعة وكيف تُجاب بنزاهة، ومقترحات التطوير.", "معرفة الرمز النهائي بنودًا وأوزانًا والتقييم الذاتي عليه."],
    challenges_ar=["عرض الدقة بلا خط أساس.", "الدفاع عن النموذج بدل مناقشته."],
)

LESSON_MODULES = ["overview", "presentation_discussion", "rubric"]
