from core.models import Module

MODULE = Module(
    id="foundations.prob",
    section="foundations",
    title_ar="الوحدة 6 — الاحتمال والإحصاء",
    title_en="Module 6 — Probability & Statistics",
    order=6,
    icon=":material/bar_chart:",
    prerequisites=["foundations.math"],
    purpose_ar="المتوسط والتباين والتوزيعات والاحتمال الشرطي والارتباط، ثم الاحتمال الأرجح والإنتروبيا: الأساس الذي تُشتق منه دوال خسارة التصنيف ومفاهيم التعميم.",
    why_ar="الإنتروبيا المتقاطعة ليست صيغة اعتباطية: هي سالب لوغاريتم الاحتمال الأرجح. من يفهم هذا يعرف لماذا Softmax + CCE زوج طبيعي، ولماذا التحجيم يستخدم المتوسط والانحراف.",
    objectives_ar=[
        "حساب المتوسط والتباين والانحراف المعياري وربطها بالتحجيم.",
        "قراءة توزيع، احتمال، احتمال شرطي، توقّع، وعينة.",
        "تفسير الارتباط والتغاير وحدودهما.",
        "اشتقاق خسارة التصنيف من الاحتمال الأرجح وفهم الإنتروبيا والإنتروبيا المتقاطعة.",
    ],
    challenges_ar=["الخلط بين الاحتمال والاحتمال الأرجح.", "الاعتقاد أن الارتباط سببية.", "الخوف من الإنتروبيا رغم أنها «متوسط المفاجأة»."],
)

LESSON_MODULES = ["descriptive", "distributions", "correlation", "likelihood", "entropy_cross_entropy"]
