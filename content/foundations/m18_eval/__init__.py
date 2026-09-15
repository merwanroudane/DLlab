from core.models import Module

MODULE = Module(
    id="foundations.eval",
    section="foundations",
    title_ar="الوحدة 18 — تقييم النموذج",
    title_en="Module 18 — Model Evaluation",
    order=18,
    icon=":material/fact_check:",
    prerequisites=["foundations.training_loop", "foundations.ml"],
    purpose_ar="مقاييس التدريب/التحقق/الاختبار، الدقة والصحة والاستدعاء وF1 ومصفوفة الالتباس، ROC/AUC والعتبة، مقاييس الانحدار، واختيار المقياس — ولماذا تضلل الدقة.",
    why_ar="النموذج لا يُحكم عليه بخسارته بل بمقياس يعكس القرار الاقتصادي/الإداري الذي سيخدمه. اختيار المقياس الخاطئ يعني تحسين الشيء الخاطئ.",
    objectives_ar=["حساب كل مقياس من مصفوفة الالتباس يدويًا.", "قراءة ROC وحساب AUC وفهم معنى العتبة كقرار كلفة.", "اختيار المقياس بحسب المهمة وعدم التوازن والكلفة."],
    challenges_ar=["الاعتماد على الدقة في بيانات غير متوازنة.", "الخلط بين الصحة والاستدعاء.", "قراءة R² كنسبة صواب."],
)

LESSON_MODULES = ["splits_metrics", "classification_metrics", "roc_auc_threshold", "regression_metrics", "metric_selection"]
