from core.models import Module

MODULE = Module(
    id="foundations.optim",
    section="foundations",
    title_ar="الوحدة 15 — التحسين",
    title_en="Module 15 — Optimization",
    order=15,
    icon=":material/speed:",
    prerequisites=["foundations.backprop", "foundations.calculus"],
    purpose_ar="سطح الخسارة وحدوده الدنيا ونقاط السرج، ثم الانحدار التدريجي بأنواعه، معدل التعلم ومشكلاته، الزخم وRMSprop وAdam، مقارنة المحسّنات، جداول معدل التعلم، والتقارب.",
    why_ar="نفس الشبكة ونفس البيانات تنجح أو تفشل حسب المحسّن ومعدل التعلم. هذه الوحدة تحوّل «جرّب Adam» إلى فهم لماذا.",
    objectives_ar=["قراءة سطح الخسارة: حد أدنى محلي/عام، سرج، انحناء، محدب/غير محدب.", "التمييز بين Batch/SGD/Mini-batch وأثر حجم الدفعة على الضوضاء.", "تشخيص معدل التعلم الصغير/الكبير/المتذبذب/المتباعد/NaN.", "شرح Momentum وRMSprop وAdam معادلاتٍ وحدسًا، واختيار بينها."],
    challenges_ar=["الاعتقاد أن Adam يغني عن ضبط معدل التعلم.", "الخلط بين التذبذب بسبب الدفعات الصغيرة والتذبذب بسبب معدل كبير.", "توقع الوصول إلى الحد الأدنى العام."],
)

LESSON_MODULES = ["loss_surface", "gd_variants", "learning_rate", "momentum_nesterov", "rmsprop_adam", "optimizer_comparison", "schedules_convergence"]
