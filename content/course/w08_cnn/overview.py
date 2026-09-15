from components.quiz import Q
from components.week import week_overview
from core.models import Lesson

LESSON = Lesson(
    id="course.w08.overview",
    title_ar="نظرة عامة على الأسبوع 08 والاختبار القبلي",
    title_en="Week 08 Overview & Pre-test",
    module="course.w08",
    order=1,
    prerequisites=["course.w07.diagnose_interpret", "foundations.linalg.tensors", "foundations.data.shape_axis_rank"],
    objectives_ar=["خريطة الأسبوع وروابط الأسس.", "اختبار قبلي على الموترات والالتفاف."],
    terms=["tensor", "channel_dimension", "shape"],
    difficulty="beginner",
    summary_ar="الصورة كموتر ← النواة والالتفاف وخريطة الخصائص ← الحشو والخطوة والتجميع ← الطبقات والبنى وحساب الأشكال.",
)


def render() -> None:
    week_overview(
        LESSON, week_no="08",
        steps=["Image as tensor", "Kernel & convolution", "Feature maps", "Padding · stride · pooling", "CNN layers & architectures", "Shape calculation", "Post-test"],
        links=[("الصورة كموتر، البكسل، القناة", "الالتفاف", "الأسس 1 و4: الشكل والمحاور، بُعد القناة"), ("النواة/المرشّح، الالتفاف، خريطة الخصائص، تحريك النواة", "الالتفاف", "الأسس 4: الضرب العنصري والجمع؛ الأسس 9: المجموع الموزون"),
               ("الحشو، الخطوة، التجميع", "الحشو والخطوة والتجميع", "معمل الحشو والخطوة"), ("الطبقات الأساسية والبنى النموذجية وحساب الأشكال", "الطبقات والأشكال", "الأسس 10: عدّ المعلمات؛ الوحدة 21: summary")],
        buttons=[("معمل الالتفاف (النواة المتحركة)", ":material/science:", "labs.cnn_convolution_lab"), ("حاسبة أشكال CNN", ":material/science:", "labs.cnn_shape_calculator"), ("الأسس 4 — الموترات", ":material/menu_book:", "foundations.linalg.tensors")],
        pretest=[Q("صورة ملونة 32×32 كموتر (Keras):", ["(32, 32)", "(32, 32, 3)", "(3, 32)"], 1, ""), Q("نواة 3×3 على قناة واحدة: معلماتها", ["9", "10", "3"], 1, "9 + انحياز"),
                 Q("الالتفاف ينتج…", ["رقمًا واحدًا", "خريطة خصائص", "متجه أوزان"], 1, ""), Q("MaxPooling(2) على 28×28 يعطي", ["28×28", "14×14", "26×26"], 1, "")],
        note_ar="أول بنية غير MLP: كل ما تحتاجه هو الجبر الخطي (الضرب العنصري والجمع) وحساب الأشكال. المعامل الثلاثة تجعل النواة تتحرك أمام عينيك.",
        takeaway_ar="الأسبوع 08 = الصورة موتر، النواة كاشف نمط متحرك، والأشكال تُحسب بصيغة واحدة.")
