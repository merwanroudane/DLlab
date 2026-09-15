from components.quiz import Q
from components.week import week_overview
from core.models import Lesson

LESSON = Lesson(
    id="course.w13.overview",
    title_ar="نظرة عامة على الأسبوع 13 والاختبار القبلي",
    title_en="Week 13 Overview & Pre-test",
    module="course.w13",
    order=1,
    prerequisites=["course.w12.applications", "foundations.frameworks.keras.errors"],
    objectives_ar=["خريطة الأسبوع.", "اختبار قبلي على GPU والذاكرة."],
    terms=["batch_size"],
    difficulty="beginner",
    summary_ar="CPU مقابل GPU والتوازي ← VRAM وحجم الدفعة ← Colab ووقت التشغيل ← اكتشاف GPU ومعلومات الجهاز ← OOM والأداء.",
)


def render() -> None:
    week_overview(
        LESSON, week_no="13",
        steps=["CPU vs GPU", "Parallelism", "VRAM & batch size", "Colab runtime", "GPU detection", "OOM diagnostics", "Performance notes", "Post-test"],
        links=[("CPU مقابل GPU، التوازي، VRAM، الدفعة مقابل الذاكرة", "CPU وGPU والذاكرة", "الأسس 17: الدفعة؛ الأسس 4: ضرب المصفوفات"), ("Colab، وقت التشغيل، الاكتشاف، معلومات الجهاز، OOM، الأداء", "Colab ووقت التشغيل", "الوحدة 21: الأجهزة والأخطاء؛ الوحدة 22: لقطات Colab/GPU")],
        buttons=[("معمل GPU وذاكرة الدفعة", ":material/science:", "labs.gpu_batch_memory_lab"), ("الوحدة 22 — معرض PyTorch (CPU/GPU)", ":material/photo_library:", "foundations.gallery.pytorch")],
        pretest=[Q("GPU يسرّع التدريب لأنه…", ["أدق", "ينفذ آلاف عمليات الضرب بالتوازي", "يملك ذاكرة أكبر"], 1, ""), Q("`ResourceExhaustedError: OOM`:", ["خطأ في الكود", "ذاكرة GPU امتلأت: قلّل الدفعة", "GPU غير موجود"], 1, ""),
                 Q("GPU يغيّر…", ["الدقة النهائية", "السرعة (وأحيانًا الكسور العشرية)", "الخسارة"], 1, ""), Q("في Colab، لتفعيل GPU:", ["pip install gpu", "Runtime → Change runtime type → GPU", "يُفعَّل تلقائيًا"], 1, "")],
        note_ar="على هذه الآلة لا GPU؛ الأمثلة تعمل على CPU والقياسات تُظهر الحدس. في Colab شغّل نفس الملفات بلا تعديل.",
        takeaway_ar="الأسبوع 13 = لماذا GPU (توازٍ)، حدوده (VRAM)، وكيف تحصل عليه وتشخّصه في Colab.")
