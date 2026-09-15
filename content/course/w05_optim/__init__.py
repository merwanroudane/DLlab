from core.models import Module

MODULE = Module(
    id="course.w05",
    section="course",
    title_ar="الأسبوع 05 — خوارزميات التحسين",
    title_en="Week 05 — Optimization Algorithms",
    order=5,
    icon=":material/looks_5:",
    prerequisites=["course.w04", "foundations.optim", "foundations.loss"],
    purpose_ar="الانحدار التدرجي، معدل التعلم، SGD، الزخم، RMSprop، Adam، أثر دالة الخسارة على التدريب، سطح الخسارة، التقارب، مشاكل معدل التعلم، وسباق المحسّنات التفاعلي — مع تطبيق كل ذلك في Keras على نموذج حقيقي.",
    why_ar="المحسّن ومعدل التعلم هما أول ما يُشخَّص عندما لا يتعلم النموذج. هذا الأسبوع يجعل السلوكيات (تذبذب، بطء، انفجار، هضبة) مرئية ثم يربطها بمعاملات compile.",
    objectives_ar=["شرح خطوة الانحدار التدرجي ودور η وأثره الثلاثي (صغير/مناسب/كبير).", "المقارنة بين SGD والزخم وRMSprop وAdam على سطح خسارة وعلى شبكة حقيقية.", "تشخيص مشاكل معدل التعلم من منحنى الخسارة واختيار جدول."],
    challenges_ar=["اعتبار Adam حلًا لكل شيء بلا ضبط η.", "قراءة تذبذب الخسارة كخطأ في البيانات."],
)

LESSON_MODULES = ["overview", "gd_lr_sgd", "momentum_rmsprop_adam", "loss_surface_convergence"]
