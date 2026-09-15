from core.models import Module

MODULE = Module(
    id="foundations.math",
    section="foundations",
    title_ar="الوحدة 3 — الأسس الرياضية",
    title_en="Module 3 — Mathematical Foundations",
    order=3,
    icon=":material/functions:",
    prerequisites=["foundations.python"],
    purpose_ar="اللغة الرياضية الدنيا لقراءة معادلات الشبكات: الرموز، الدوال، الأسس واللوغاريتمات، رمز المجموع، المتوسط والقيمة المطلقة.",
    why_ar="معادلة مثل $L = -\\frac{1}{n}\\sum_i y_i \\log \\hat{y}_i$ تجمع أربعة مفاهيم من هذه الوحدة. من دونها تبقى طلاسم.",
    objectives_ar=[
        "قراءة المتغير والثابت والمعامل والرمز الفرعي في معادلة.",
        "فهم الدالة كآلة مدخل ← مخرج وتمييز الخطي من اللاخطي بيانيًا.",
        "استخدام الأس واللوغاريتم الطبيعي وفهم لماذا يظهران في Softmax والإنتروبيا المتقاطعة.",
        "قراءة رمز المجموع Σ وترجمته إلى حلقة أو إلى `np.sum`.",
    ],
    challenges_ar=["الخوف من الرموز.", "الخلط بين $x_i$ (ملاحظة) و$x^2$ (أس).", "نسيان أن log(0) غير معرّف."],
)

LESSON_MODULES = ["notation", "functions", "exponent_log", "summation_average_abs"]
