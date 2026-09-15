from components.quiz import Q
from components.week import week_overview
from core.models import Lesson

LESSON = Lesson(
    id="course.w14.overview",
    title_ar="نظرة عامة على الأسبوع 14 والاختبار القبلي",
    title_en="Week 14 Overview & Pre-test",
    module="course.w14",
    order=1,
    prerequisites=["course.w13.colab_runtime", "course.w07.diagnose_interpret"],
    objectives_ar=["خريطة المشروع بمراحله السبع عشرة (§81).", "اختبار قبلي على منهج المشروع."],
    terms=["reproducibility", "seed"],
    difficulty="beginner",
    summary_ar="تعريف ← بيانات ← تشخيص ← معالجة ← تقسيم ← فحص التسريب ← خط أساس ← بنية ← معلمات فائقة ← تدريب ← منحنيات ← تشخيص ← إصلاح ← تقييم ← تفسير ← استنساخ ← عرض.",
)


def render() -> None:
    week_overview(
        LESSON, week_no="14",
        steps=["Problem", "Dataset", "Diagnostics", "Preprocessing", "Split", "Leakage check", "Baseline", "Architecture", "Hyperparameters", "Training", "Curves", "Diagnostics", "Fixing", "Evaluation", "Interpretation", "Reproducibility", "Presentation"],
        links=[("تعريف المسألة ووصف البيانات وتشخيصها", "دليل المشروع", "الأسس 1 و8؛ الأسبوع 02"), ("المعالجة والتقسيم وخط الأساس", "دليل المشروع", "الأسبوعان 02 و07"), ("البنية والمعلمات الفائقة والتدريب والتحقق", "دليل المشروع", "الأسابيع 03–06؛ 08–12 بحسب نوع البيانات"),
               ("التشخيص والتقييم والتفسير والاستنساخ", "دليل المشروع + قالب الدفتر", "الأسبوع 07؛ الأسس 18–20؛ الأسبوع 13 (Colab)")],
        buttons=[("الأسبوع 07 — القالب المصغّر", ":material/menu_book:", "course.w07"), ("الوحدة 23 — حالة الجاهزية", ":material/verified:", "foundations.ready.ready_status")],
        pretest=[Q("أول قرار في المشروع:", ["البنية", "تعريف المسألة والهدف ووحدته", "المحسّن"], 1, ""), Q("خط الأساس…", ["اختياري", "إلزامي ويُبلَّغ بجانب النموذج", "للانحدار فقط"], 1, ""),
                 Q("الاستنساخ يتطلب…", ["دقة عالية", "بذرة ونسخًا وكودًا يعيد كل الأرقام", "GPU"], 1, ""), Q("مجموعة الاختبار…", ["تُستخدم لضبط η", "تُلمس مرة واحدة في النهاية", "تُخلط مع التحقق"], 1, "")],
        note_ar="اختر مسألة تملك بياناتها الآن (لا «سأجمعها»). جدول بمئات الصفوف كافٍ لمشروع ممتاز إن كان المنهج سليمًا.",
        takeaway_ar="الأسبوع 14 = المنهج كاملًا على بياناتك، بقالب دفتر ورمز تقييم معلن.")
