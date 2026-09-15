from core.models import Module

MODULE = Module(
    id="foundations.prep",
    section="foundations",
    title_ar="الوحدة 8 — إعداد البيانات للتعلم الآلي والعميق",
    title_en="Module 8 — Data Preparation for ML/DL",
    order=8,
    icon=":material/cleaning_services:",
    prerequisites=["foundations.ml", "foundations.data"],
    purpose_ar="الفحص والتنظيف، الترميز، التحجيم، التقسيم، التسريب، عدم التوازن والزيادة، ثم التحويل إلى موترات ودفعات وخطوط أنابيب.",
    why_ar="أغلب فشل النماذج يبدأ هنا لا في البنية: تسريب صامت يعطي دقة خيالية، أو تحجيم خاطئ يجعل التدريب مستحيلًا. ترتيب التشخيص يبدأ من البيانات.",
    objectives_ar=[
        "فحص وتنظيف جدول: مفقودات، تكرار، شواذ، تسميات خاطئة.",
        "ترميز الفئات وتحجيم الأعداد بالترتيب الصحيح (fit على التدريب فقط).",
        "تقسيم طبقي وزمني صحيح، واكتشاف التسريب بأنواعه الثلاثة.",
        "التعامل مع عدم التوازن، وتحويل البيانات إلى موترات ودفعات.",
    ],
    challenges_ar=["fit_transform على كل البيانات قبل التقسيم.", "خاصية تُعرف بعد الحدث.", "حذف المفقودات بلا تفكير في سببها."],
)

LESSON_MODULES = ["inspection_cleaning", "encoding", "scaling", "splitting", "leakage", "imbalance_augmentation", "tensors_batching_pipelines"]
