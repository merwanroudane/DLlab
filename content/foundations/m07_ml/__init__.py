from core.models import Module

MODULE = Module(
    id="foundations.ml",
    section="foundations",
    title_ar="الوحدة 7 — أسس التعلم الآلي",
    title_en="Module 7 — Machine Learning Foundations",
    order=7,
    icon=":material/psychology:",
    prerequisites=["foundations.prob", "foundations.calculus"],
    purpose_ar="أنماط التعلم وأنواع المسائل، ثم الجسر من الانحدار الخطي واللوجستي إلى الخلية العصبية: نفس المعادلات، نفس التدريب، تنشيط مختلف.",
    why_ar="الانحدار اللوجستي **هو** خلية عصبية واحدة. من يفهمه يفهم الشبكة كتراكب لهذه الخلايا؛ ومن يقفز فوقه يرى الشبكة صندوقًا أسود.",
    objectives_ar=[
        "تصنيف أي مسألة: نمط التعلم (مُشرف/غير مُشرف...) ونوع المهمة (انحدار/تصنيف...).",
        "التمييز بين الهدف والخسارة والمقياس وخط الأساس والتعميم.",
        "تدريب انحدار خطي ولوجستي بالانحدار التدريجي من الصفر وقراءة معلماتهما.",
    ],
    challenges_ar=["الخلط بين الخسارة (للتدريب) والمقياس (للتقييم).", "نسيان خط الأساس.", "الاعتقاد أن التصنيف ينتج فئة مباشرة لا احتمالًا."],
)

LESSON_MODULES = ["paradigms", "task_types", "objective_loss_metric", "baseline_generalization", "linear_regression", "logistic_regression"]
