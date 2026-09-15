from core.models import Module

MODULE = Module(
    id="foundations.ready",
    section="foundations",
    title_ar="الوحدة 23 — جاهز للتعلم العميق",
    title_en="Module 23 — Ready for Deep Learning",
    order=23,
    icon=":material/verified:",
    prerequisites=["foundations.frameworks", "foundations.gallery"],
    purpose_ar="خريطة المفاهيم وخريطة الأطر، فحص المتطلبات، اختبار تشخيصي، ست تحديات عملية (الأشكال، الحقب/الدفعات، الخسارة/المحسّن، سير عمل Keras، موترات/اشتقاق TensorFlow، حلقة PyTorch)، وحالة الجاهزية المحسوبة من تقدمك ونتائجك.",
    why_ar="قبل المقرر الرسمي تحتاج دليلًا لا شعورًا: ما الذي تتقنه فعلًا وما الذي يجب أن تعود إليه. هذه الوحدة تقيس وتوجّه.",
    objectives_ar=["رؤية كل ما تعلمته كخريطة مترابطة.", "اكتشاف الفجوات بالاختبار والتحديات لا بالانطباع.", "الحصول على حالة جاهزية بقائمة عودة محددة."],
    challenges_ar=["الاكتفاء بالقراءة دون حل التحديات.", "اعتبار درجة واحدة منخفضة حكمًا نهائيًا بدل مؤشر للعودة."],
)

LESSON_MODULES = ["concept_map", "prerequisite_check", "challenges", "ready_status"]
