from core.models import Module

MODULE = Module(
    id="foundations.regularization",
    section="foundations",
    title_ar="الوحدة 20 — التنظيم",
    title_en="Module 20 — Regularization",
    order=20,
    icon=":material/shield:",
    prerequisites=["foundations.generalization"],
    purpose_ar="L1 وL2/تضاؤل الأوزان، Dropout، الإيقاف المبكر، زيادة البيانات وتبسيط النموذج، وBatch Normalization بشرح دقيق: كيف يعمل كل منها ولماذا ومتى.",
    why_ar="التنظيم هو ما يحوّل شبكة تحفظ إلى شبكة تعمّم. لكنه ليس مجانيًا: كل أداة لها كلفة وسلوك مختلف في التدريب والاستدلال.",
    objectives_ar=["اشتقاق أثر L2 على قاعدة التحديث (تضاؤل الأوزان) والفرق عن L1.", "شرح Dropout في التدريب والاستدلال (القلب المعكوس) وتجربته.", "فهم Batch Normalization: ماذا يطبّع، معلماته γ وβ، وسلوك train/eval."],
    challenges_ar=["Dropout مفعّل أثناء الاستدلال.", "BatchNorm مع دفعة صغيرة.", "التنظيم كحل لقصور التعلم."],
)

LESSON_MODULES = ["l1_l2_weight_decay", "dropout", "early_stopping", "augmentation_simplification", "batch_normalization"]
