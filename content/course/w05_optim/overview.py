from components.quiz import Q
from components.week import week_overview
from core.models import Lesson

LESSON = Lesson(
    id="course.w05.overview",
    title_ar="نظرة عامة على الأسبوع 05 والاختبار القبلي",
    title_en="Week 05 Overview & Pre-test",
    module="course.w05",
    order=1,
    prerequisites=["course.w04.depth_width", "foundations.optim.learning_rate", "foundations.optim.rmsprop_adam"],
    objectives_ar=["خريطة الأسبوع وروابط الأسس.", "اختبار قبلي على الانحدار التدرجي والمحسّنات."],
    terms=["gradient", "learning_rate", "optimizer"],
    difficulty="beginner",
    summary_ar="GD ← η ← SGD ← الزخم ← RMSprop ← Adam ← الخسارة والسطح ← التقارب ومشاكل η ← سباق المحسّنات.",
)


def render() -> None:
    week_overview(
        LESSON, week_no="05",
        steps=["Gradient descent & η", "SGD & batches", "Momentum · RMSprop · Adam", "Loss surface & convergence", "LR problems & schedules", "Optimizer race", "Post-test"],
        links=[("الانحدار التدرجي ومعدل التعلم", "GD وη وSGD", "الأسس 15: معدل التعلم، GD وأنواعه"), ("SGD والدفعات", "GD وη وSGD", "الأسس 17: الدفعة والحقبة"), ("الزخم وRMSprop وAdam", "الزخم وRMSprop وAdam", "الأسس 15: الزخم، RMSprop/Adam"),
               ("الخسارة وأثرها، سطح الخسارة، التقارب", "سطح الخسارة والتقارب", "الأسس 13، 15: سطح الخسارة، الجداول والتقارب"), ("سباق المحسّنات", "الزخم وRMSprop وAdam", "معمل سباق المحسّنات")],
        buttons=[("الأسس 15 — التحسين", ":material/menu_book:", "foundations.optim"), ("معمل سباق المحسّنات", ":material/science:", "labs.optimizer_race"), ("معمل معدل التعلم", ":material/science:", "labs.learning_rate_lab")],
        pretest=[Q("خطوة الانحدار التدرجي:", ["θ ← θ + η∇L", "θ ← θ − η∇L", "θ ← ∇L"], 1, ""), Q("η كبير جدًا يسبب…", ["تقاربًا بطيئًا", "تذبذبًا أو انفجارًا", "لا شيء"], 1, ""),
                 Q("الزخم يساعد على…", ["تسريع الاتجاهات المتسقة وتخفيف التذبذب", "زيادة الضوضاء", "تقليل المعلمات"], 0, ""), Q("Adam يجمع بين…", ["الزخم وتكييف η لكل معلمة", "L2 وDropout", "batch وepoch"], 0, "")],
        note_ar="الأسس 15 تشرح كل محسّن رياضيًا وبالتحريك؛ هذا الأسبوع يشغّلها على شبكة Keras حقيقية ويقرأ منحنيات الخسارة كطبيب.",
        takeaway_ar="الأسبوع 05 = السلوك المرئي لكل محسّن ومعدل تعلم، ثم تطبيقه في compile وقراءة منحنى الخسارة.")
