from components.quiz import Q
from components.week import week_overview
from core.models import Lesson

LESSON = Lesson(
    id="course.w07.overview",
    title_ar="نظرة عامة على الأسبوع 07 والاختبار القبلي",
    title_en="Week 07 Overview & Pre-test",
    module="course.w07",
    order=1,
    prerequisites=["course.w06.choosing_activations", "course.w02.evaluation_baselines"],
    objectives_ar=["خريطة المشروع المصغّر وروابط الأسس.", "اختبار قبلي على منهج المشروع."],
    terms=["dataset", "feature", "target"],
    difficulty="beginner",
    summary_ar="تعريف المسألة ← البيانات وإعدادها ← خط الأساس ← البنية ← التدريب ← التقييم ← التشخيص ← التفسير ← تحليل الأخطاء ← التقرير.",
)


def render() -> None:
    week_overview(
        LESSON, week_no="07",
        steps=["Problem", "Data prep", "Baseline", "Architecture", "Training", "Evaluation", "Diagnostics", "Interpretation", "Error analysis", "Post-test"],
        links=[("تعريف المسألة والبيانات وإعدادها", "بناء خط الأنابيب", "الأسس 1 و8؛ الأسبوع 02"), ("خط الأساس والبنية والتدريب", "بناء خط الأنابيب", "الأسبوعان 02 و04؛ الأسس 20 (الإيقاف المبكر)"),
               ("التقييم والتشخيص", "التشخيص والتفسير", "الأسس 18 و19؛ الأسبوع 05 (المنحنيات)"), ("التفسير وتحليل الأخطاء", "التشخيص والتفسير", "الأسبوع 01 (تنبؤ ≠ سببية)"), ("التقرير والاستنساخ", "التشخيص والتفسير", "الأسس 2 (البذرة)")],
        buttons=[("الأسس 8 — إعداد البيانات", ":material/menu_book:", "foundations.prep"), ("الأسس 19 — التعميم", ":material/menu_book:", "foundations.generalization"), ("معمل تشخيص المنحنيات", ":material/science:", "labs.curves_diagnostic_lab")],
        pretest=[Q("أول نموذج تبنيه في مشروع:", ["أعمق شبكة", "خط أساس بسيط", "LSTM"], 1, ""), Q("RMSE يُبلَّغ…", ["كنسبة", "بوحدة الهدف", "كخسارة"], 1, ""),
                 Q("val_loss يرتفع من الحقبة 5 بينما loss ينخفض:", ["قصور", "فرط تخصيص → إيقاف مبكر/تنظيم", "تسريب"], 1, ""), Q("أهمية خاصية عالية في النموذج تعني…", ["سببية", "أن النموذج يعتمد عليها للتنبؤ", "أنها مقاسة جيدًا"], 1, "")],
        note_ar="هذا الأسبوع قالب مشروع الأسبوع 14: نفّذه على بيانات المنصة هنا، ثم على بياناتك هناك.",
        takeaway_ar="الأسبوع 07 = المنهج كاملًا على مسألة واحدة، بقرارات مبرَّرة وأرقام على بيانات محجوزة.")
