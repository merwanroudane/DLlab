from core.models import Module

MODULE = Module(
    id="foundations.loss",
    section="foundations",
    title_ar="الوحدة 13 — الخطأ والخسارة والهدف",
    title_en="Module 13 — Error, Loss & Objective",
    order=13,
    icon=":material/trending_down:",
    prerequisites=["foundations.forward", "foundations.prob"],
    purpose_ar="من الخطأ الفردي إلى الخسارة الكلية: MSE وMAE وHuber للانحدار، BCE للثنائي، CCE وSparse CCE للمتعدد، وقاعدة توافق الخسارة مع تنشيط الإخراج.",
    why_ar="الخسارة هي ما يقلله التدريب فعلًا. خسارة خاطئة أو غير متوافقة مع الإخراج تعني أن الشبكة تتعلم الشيء الخطأ بثقة.",
    objectives_ar=["حساب كل خسارة يدويًا وبالكود لأمثلة صغيرة.", "اختيار الخسارة من نوع الهدف وشكله.", "تشخيص عدم التوافق بين الخسارة وتنشيط الإخراج."],
    challenges_ar=["Softmax + BCE.", "أرقام فئات مع CCE (يتوقع one-hot).", "MSE للتصنيف."],
)

LESSON_MODULES = ["error_loss_cost", "mse_mae_huber", "bce", "cce_sparse", "loss_activation_compatibility"]
