from core.models import Module

MODULE = Module(
    id="foundations.data",
    section="foundations",
    title_ar="الوحدة 1 — أسس البيانات",
    title_en="Module 1 — Data Foundations",
    order=1,
    icon=":material/table_chart:",
    prerequisites=["foundations.start"],
    purpose_ar=(
        "بناء المفردات الأساسية للبيانات كما تراها الشبكة العصبية: الملاحظة، الخاصية، الهدف، "
        "أنواع البيانات والمتغيرات، ثم `dtype` و`shape` و`axis` و`rank` — المفاهيم التي تسبب "
        "معظم أخطاء التشغيل الأولى."
    ),
    why_ar=(
        "أول خطأ يقابله الباحث في `Keras` أو `PyTorch` يكون غالبًا `ValueError: Shapes ... are incompatible` "
        "أو خطأ `dtype`. هذه الأخطاء ليست أخطاء «شبكات» بل أخطاء فهم للبيانات وتمثيلها."
    ),
    objectives_ar=[
        "التمييز بين الملاحظة والخاصية والهدف في أي مجموعة بيانات.",
        "تصنيف البيانات (جدولية، زمنية، نصية، صور) والمتغيرات (عددية، فئوية...).",
        "قراءة `shape` و`ndim` و`dtype` وتفسيرها بثقة.",
        "تشريح مجموعة بيانات حقيقية في المعمل وتحويلها إلى `X` و`y`.",
    ],
    challenges_ar=[
        "الخلط بين محور الملاحظات ومحور الخصائص.",
        "الاعتقاد أن `shape` تفصيل برمجي غير مهم.",
        "الخلط بين النوع الإحصائي للمتغير (فئوي/عددي) ونوع التخزين `dtype`.",
    ],
)

LESSON_MODULES = [
    "what_is_data",
    "dataset",
    "observation",
    "feature",
    "target",
    "data_modalities",
    "variable_types",
    "dtype",
    "shape_axis_rank",
    "dataset_anatomy",
]
