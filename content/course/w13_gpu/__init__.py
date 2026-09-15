from core.models import Module

MODULE = Module(
    id="course.w13",
    section="course",
    title_ar="الأسبوع 13 — GPU وGoogle Colab",
    title_en="Week 13 — GPU & Google Colab",
    order=13,
    icon=":material/memory:",
    prerequisites=["course.w12", "foundations.frameworks"],
    purpose_ar="تسريع النماذج بـ GPU: CPU مقابل GPU، حدس التوازي، ذاكرة GPU (VRAM)، حجم الدفعة مقابل الذاكرة، Google Colab واختيار وقت التشغيل، اكتشاف GPU، معلومات الجهاز في TensorFlow، تشخيص OOM، وملاحظات الأداء.",
    why_ar="مشاريع الصور والتسلسلات الحقيقية لا تُدرَّب على CPU في وقت معقول. GPU لا يغيّر الكود لكنه يغيّر ما يمكن تجربته — بشرط فهم الذاكرة والدفعة وOOM.",
    objectives_ar=["شرح لماذا GPU أسرع (التوازي) وما حدوده (VRAM).", "تقدير ذاكرة خطوة التدريب واختيار حجم دفعة يناسب البطاقة.", "تشغيل مشروع في Colab بـ GPU، اكتشافه من TensorFlow/PyTorch، وتشخيص OOM."],
    challenges_ar=["توقع أن GPU يحسّن الدقة.", "اختيار أكبر دفعة ممكنة دون تعديل η."],
)

LESSON_MODULES = ["overview", "cpu_gpu_memory", "colab_runtime"]
