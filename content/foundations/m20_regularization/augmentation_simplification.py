import streamlit as st

from components.callouts import common_mistake, practical_note, research_note, takeaway
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.regularization.augmentation_simplification",
    title_ar="زيادة البيانات وتبسيط النموذج: خريطة التنظيم الكاملة",
    title_en="Data Augmentation & Model Simplification: the Full Regularization Map",
    module="foundations.regularization",
    order=4,
    prerequisites=["foundations.regularization.early_stopping", "foundations.prep.imbalance_augmentation"],
    objectives_ar=["ترتيب أدوات التنظيم من الأقوى إلى الأضعف ومن الأرخص إلى الأغلى.", "تبسيط النموذج كخيار أول قبل التنظيم.", "وصفة عملية للجداول والصور والتسلسلات."],
    terms=["hyperparameter"],
    difficulty="beginner",
    summary_ar="بيانات أكثر > زيادة بيانات > نموذج أصغر > إيقاف مبكر > Dropout/L2 > BN. ابدأ بالأرخص والأقوى.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("الخريطة", "The map")
    compare_table(["الأداة", "English", "الآلية", "القوة", "الكلفة", "متى"],
                  [("بيانات أكثر", "More data", "تقليل التباين مباشرة", "الأقوى", "غالبًا الأغلى", "إن أمكن، دائمًا"), ("زيادة البيانات", "Augmentation", "أمثلة جديدة صحيحة بتحويلات", "قوية (صور/نص/صوت)", "رخيصة", "الصور خاصة؛ نادرة للجداول"),
                   ("نموذج أصغر", "Simplification", "قدرة أقل = تباين أقل", "قوية", "مجانية", "أول ما تجرّبه مع بيانات قليلة"), ("إيقاف مبكر", "Early stopping", "مسافة محدودة", "متوسطة", "مجانية", "دائمًا"),
                   ("Dropout", "Dropout", "ضوضاء + تجميع", "متوسطة–قوية", "تدريب أبطأ قليلًا", "الطبقات الكثيفة الكبيرة"), ("L2 / weight decay", "Weight decay", "أوزان صغيرة", "متوسطة", "مجانية", "افتراضي صغير (1e-4)"),
                   ("Batch Normalization", "BatchNorm", "تطبيع + ضوضاء دفعة", "تنظيم جانبي", "طبقة إضافية", "شبكات عميقة/التفافية"), ("تجميع نماذج", "Ensembling", "متوسط عدة نماذج", "قوية", "×k تدريب واستدلال", "المسابقات؛ نادرًا في المقرر")],
                  ["rtl", "ltr", "rtl", "rtl", "rtl", "rtl"])
    h2("الوصفة حسب نوع البيانات", "Recipe by data type")
    compare_table(["البيانات", "ابدأ بـ", "ثم", "أخيرًا"],
                  [("جداول صغيرة", "نموذج صغير (1–2 طبقة، 32–64) + إيقاف مبكر", "L2 1e-4، Dropout 0.2", "زيادة بيانات غير معتادة؛ فكر في نماذج شجرية للمقارنة"),
                   ("صور", "زيادة بيانات (قلب، قص، إضاءة) + إيقاف مبكر", "Dropout في الكثيفة، BN في الالتفافية", "نقل التعلم (تعمّق)"),
                   ("تسلسلات/نص", "إيقاف مبكر + Dropout في التكرارية", "L2 صغير، قصّ التدرج", "زيادة بيانات نصية بحذر")],
                  ["rtl", "rtl", "rtl", "rtl"])
    research_note("التنظيم يعالج التباين لا الانحياز. إن كان نموذجك يقصر في التدريب أصلًا فالتنظيم يزيد الأمر سوءًا. شخّص أولًا (الوحدة 19) ثم نظّم.")
    practical_note("قاعدة الترتيب: **الأرخص أولًا**. تصغير النموذج والإيقاف المبكر مجانيان؛ جرّبهما قبل ضبط λ وp لساعات.")
    common_mistake("تكديس كل الأدوات معًا من البداية (Dropout 0.5 + L2 كبير + نموذج صغير): قصور تعلم لا يمكن تشخيصه. أضف أداة واحدة، قس، ثم أضف.")
    quiz("reg.map", [
        Q("العلاج الأقوى لفرط التخصيص عمومًا…", ["Dropout", "بيانات أكثر", "L2"], 1, "تباين أقل مباشرة."),
        Q("مع جدول من 300 صف، أول ما تجرّبه…", ["شبكة أعمق", "نموذج أصغر + إيقاف مبكر", "BN"], 1, "مجاني وفعال."),
        Q("التنظيم مع قصور تعلم…", ["يحله", "يزيده سوءًا", "لا أثر"], 1, "يعالج التباين فقط."),
    ])
    takeaway("بيانات > زيادة > تبسيط > إيقاف مبكر > Dropout/L2 > BN. الأرخص أولًا، أداة واحدة في كل مرة، والتشخيص قبل العلاج.")
    lesson_footer(LESSON, ["ثماني أدوات مرتبة.", "وصفة لكل نوع بيانات.", "لا تنظيم للقصور."])
