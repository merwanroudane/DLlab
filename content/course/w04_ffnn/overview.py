from components.quiz import Q
from components.week import week_overview
from core.models import Lesson

LESSON = Lesson(
    id="course.w04.overview",
    title_ar="نظرة عامة على الأسبوع 04 والاختبار القبلي",
    title_en="Week 04 Overview & Pre-test",
    module="course.w04",
    order=1,
    prerequisites=["course.w03.pytorch_equivalent", "foundations.architecture.layers", "foundations.forward.layer_by_layer"],
    objectives_ar=["خريطة الأسبوع وروابط الأسس.", "اختبار قبلي على التدفق الأمامي والمعلمات."],
    terms=["weight", "bias", "parameter"],
    difficulty="beginner",
    summary_ar="التغذية الأمامية ← MLP ← الطبقات ← الأوزان/الانحيازات ← التمرير الأمامي ← نماذج متعددة الطبقات ← عدّ المعلمات ← التدفق المرئي.",
)


def render() -> None:
    week_overview(
        LESSON, week_no="04",
        steps=["Feedforward concept", "MLP layers", "Forward pass & shapes", "Deeper models in Keras", "Parameter count", "Depth vs width experiment", "Post-test"],
        links=[("التغذية الأمامية وMLP والطبقات", "التدفق الأمامي", "الأسس 10: طبقات ومعلمات"), ("الأوزان والانحيازات والتمرير الأمامي", "التدفق الأمامي", "الأسس 9 و12: الخلية والتمرير الأمامي"),
               ("بناء نماذج متعددة الطبقات وعدّ المعلمات", "العمق والعرض", "الأسس 10: عدّ المعلمات؛ الوحدة 21: الطبقات والنماذج"), ("التدفق المرئي", "التدفق الأمامي", "الأسس 12: التحريك العددي")],
        buttons=[("الأسس 10 — البنية", ":material/menu_book:", "foundations.architecture"), ("الأسس 12 — التمرير الأمامي", ":material/menu_book:", "foundations.forward"), ("معمل بناء الشبكة", ":material/science:", "labs.network_builder")],
        pretest=[Q("في شبكة أمامية، المعلومات تتدفق…", ["في الاتجاهين", "من المدخل إلى المخرج فقط", "عشوائيًا"], 1, ""), Q("Dense(32) على مدخل بـ 10 خصائص: المعلمات", ["320", "352", "42"], 1, ""),
                 Q("طبقة مخفية بلا تنشيط غير خطي…", ["تزيد القدرة", "تُطوى مع التي بعدها في طبقة خطية واحدة", "تمنع التدريب"], 1, ""), Q("شكل مخرج Dense(8) لدفعة من 64:", ["(8,)", "(64, 8)", "(8, 64)"], 1, "")],
        note_ar="الأسس 9 و10 و12 هي المرجع؛ هذا الأسبوع يجمعها في Keras ويجري تجربة عمق/عرض حقيقية.",
        takeaway_ar="الأسبوع 04 = بنية MLP بالأشكال والمعلمات، ثم تجربة اختيار البنية بأدلة.")
