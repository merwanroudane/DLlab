from core.models import Module

MODULE = Module(
    id="foundations.linalg",
    section="foundations",
    title_ar="الوحدة 4 — الجبر الخطي",
    title_en="Module 4 — Linear Algebra",
    order=4,
    icon=":material/grid_on:",
    prerequisites=["foundations.math", "foundations.python"],
    purpose_ar="الأعداد والمتجهات والمصفوفات والموترات: كيف تُخزَّن البيانات والأوزان، وكيف تُحسب الطبقة كضرب مصفوفات.",
    why_ar="`Dense(64)` هي `XW + b` بمصفوفة `W` بالشكل `(d, 64)`. من يفهم ضرب المصفوفات يستطيع حساب عدد المعلمات وشكل كل طبقة بلا تخمين.",
    objectives_ar=[
        "التمييز بين عدد ومتجه ومصفوفة وموتر، وقراءة رتبته وشكله.",
        "حساب الضرب النقطي وضرب المصفوفات يدويًا وبـ NumPy، ومعرفة شرط توافق الأشكال.",
        "التمييز بين ضرب المصفوفات والضرب العنصري (هادامارد).",
        "قراءة شكل موتر للصور والسلاسل، وتحديد بُعد الدفعة والقناة.",
    ],
    challenges_ar=["الخلط بين `*` و`@`.", "نسيان أن (n, d) @ (d, k) يتطلب تطابق d.", "الخلط بين المنقولة وإعادة التشكيل."],
)

LESSON_MODULES = ["scalars_vectors", "dot_product", "matrices", "matrix_multiplication", "tensors"]
