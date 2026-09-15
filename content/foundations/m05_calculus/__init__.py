from core.models import Module

MODULE = Module(
    id="foundations.calculus",
    section="foundations",
    title_ar="الوحدة 5 — التفاضل للتعلم العميق",
    title_en="Module 5 — Calculus for Deep Learning",
    order=5,
    icon=":material/show_chart:",
    prerequisites=["foundations.math", "foundations.linalg"],
    purpose_ar="من الميل إلى المشتقة إلى المشتقة الجزئية إلى التدرج إلى قاعدة السلسلة: الأدوات التي تجعل الانتشار الخلفي مفهومًا لا سحرًا.",
    why_ar="التدريب = تحريك المعلمات في عكس اتجاه التدرج. التدرج مشتقات جزئية، وحسابه عبر الطبقات قاعدة سلسلة. هذه الوحدة هي محرك كل ما بعدها.",
    objectives_ar=[
        "فهم المشتقة كميل مماس وحسابها عدديًا ورمزيًا لدوال بسيطة.",
        "حساب مشتقة جزئية وتجميع التدرج، وتفسير اتجاهه.",
        "تطبيق قاعدة السلسلة على تركيب دوال، ثم على سلسلة حسابية (المقدمة المباشرة للانتشار الخلفي).",
        "التعرف على الجاكوبي والهسي كتعمّق اختياري.",
    ],
    challenges_ar=["الاعتقاد أن المشتقة «صيغة تُحفظ» لا معنى هندسي.", "الخلط بين اتجاه التدرج واتجاه التحديث.", "الخوف من قاعدة السلسلة رغم أنها مجرد ضرب."],
)

LESSON_MODULES = ["change_slope", "derivative", "partial_gradient", "chain_rule", "jacobian_hessian"]
