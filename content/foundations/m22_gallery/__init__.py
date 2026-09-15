from core.models import Module

MODULE = Module(
    id="foundations.gallery",
    section="foundations",
    title_ar="الوحدة 22 — معرض أطر العمل المرئي وتوجيه المنصات",
    title_en="Module 22 — Framework Visual Gallery & Platform Orientation",
    order=22,
    icon=":material/photo_library:",
    prerequisites=["foundations.frameworks"],
    purpose_ar="رؤية كيف تبدو البيئة والمخرجات فعليًا: خلية دفتر، وحدة تحكم، ملخص نموذج، سجلات تدريب، مخرجات موترات، رسوم، TensorBoard، وأخطاء نموذجية — كإعادات بناء تعليمية مبنية من مخرجات حقيقية للنسخ المثبتة، مع الأسئلة الستة لقراءة كل لقطة.",
    why_ar="الباحث الجديد يرى كتلة نص فيرتبك. المعرض يعلّمه أين الكود وأين المخرج وماذا تعني الأرقام وما الذي يجب أن يلاحظه — قبل أن يفتح Colab أو الطرفية.",
    objectives_ar=["التمييز بين بيئة التشغيل (Jupyter/Colab/IDE) وإطار العمل (بلا واجهة رسومية خاصة).", "قراءة 13 لقطة نموذجية بالأسئلة الستة.", "ربط حالات قبل/أثناء/بعد التدريب بالرياضيات ومنحنيات التعلم."],
    challenges_ar=["توقّع واجهة رسومية لـ Keras أو PyTorch (لا توجد).", "الخلط بين رسالة تحذير ورسالة خطأ.", "قراءة رقم واحد من سطر يحوي ثمانية."],
)

LESSON_MODULES = ["orientation", "gallery_keras_tf", "gallery_pytorch", "gallery_errors", "before_after"]
