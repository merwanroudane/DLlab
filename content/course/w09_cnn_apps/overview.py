from components.quiz import Q
from components.week import week_overview
from core.models import Lesson

LESSON = Lesson(
    id="course.w09.overview",
    title_ar="نظرة عامة على الأسبوع 09 والاختبار القبلي",
    title_en="Week 09 Overview & Pre-test",
    module="course.w09",
    order=1,
    prerequisites=["course.w08.layers_shapes", "course.w07.diagnose_interpret"],
    objectives_ar=["خريطة مشروع الصور المصغّر.", "اختبار قبلي على المعالجة المسبقة وفرط التخصيص في الصور."],
    terms=["tensor", "channel_dimension"],
    difficulty="beginner",
    summary_ar="معالجة الصور ← بناء CNN ← التدريب ← التقييم ← فرط التخصيص وزيادة البيانات ← خرائط الخصائص ← تصحيح الأشكال.",
)


def render() -> None:
    week_overview(
        LESSON, week_no="09",
        steps=["Image preprocessing", "CNN model", "Training", "Evaluation", "Overfitting & augmentation", "Feature maps", "Shape debugging", "Post-test"],
        links=[("معالجة الصور مسبقًا وبناء CNN والتدريب والتقييم", "البناء والتدريب", "الأسبوع 08؛ الأسس 8 (التحجيم)؛ الأسبوع 03 (Keras)"), ("فرط التخصيص وزيادة البيانات", "فرط التخصيص وزيادة البيانات", "الأسس 19 و20؛ الأسبوع 07"),
               ("خرائط الخصائص والنوى المتعلَّمة", "خرائط الخصائص والتصحيح", "الأسبوع 08: الالتفاف"), ("تصحيح أخطاء الأشكال", "خرائط الخصائص والتصحيح", "الوحدة 21: الأخطاء؛ حاسبة الأشكال")],
        buttons=[("حاسبة أشكال CNN", ":material/science:", "labs.cnn_shape_calculator"), ("الأسس 20 — التنظيم", ":material/menu_book:", "foundations.regularization")],
        pretest=[Q("قبل إدخال صور 0–255 إلى CNN:", ["لا شيء", "قسمة على 255 (0–1) وإضافة بُعد القناة", "one-hot"], 1, ""), Q("60 صورة تدريب وشبكة بـ 10 آلاف معلمة:", ["تعميم ممتاز", "فرط تخصيص محتمل → زيادة بيانات/تنظيم", "قصور"], 1, ""),
                 Q("زيادة البيانات (augmentation) تُطبَّق على…", ["التدريب فقط", "الاختبار", "الكل"], 0, ""), Q("`expected ndim=4, found ndim=3`:", ["الدفعة كبيرة", "غابت قناة أو دفعة", "الخسارة خاطئة"], 1, "")],
        note_ar="الصور هنا اصطناعية (شرائط وصلبان 16×16) لتُدرَّب في ثوانٍ على CPU؛ المنهج نفسه ينطبق على صور المنتجات أو الرسوم البيانية للأسعار في تخصصك.",
        takeaway_ar="الأسبوع 09 = مشروع صور كامل: تجهيز، CNN، تدريب، تقييم، تشخيص، زيادة بيانات، تفسير، وتصحيح.")
