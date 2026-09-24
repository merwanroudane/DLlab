import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, practical_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from content.course.w13_gpu._viz import parallel_svg
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline, table

LESSON = Lesson(
    id="course.w13.overview",
    title_ar="نظرة عامة على الأسبوع 13 والاختبار القبلي",
    title_en="Week 13 Overview & Pre-test",
    module="course.w13",
    order=1,
    prerequisites=["course.w12.applications", "foundations.frameworks.keras.errors"],
    objectives_ar=["رؤية لماذا يسرّع GPU التعلم العميق في أربع لقطات (تحريك).", "خريطة الأسبوع.", "اختبار قبلي على GPU والذاكرة."],
    terms=["gpu", "oom", "batch_size"],
    difficulty="intermediate",
    summary_ar="CPU مقابل GPU والتوازي ← ذاكرة VRAM وحجم الدفعة ← Colab: التفعيل والتحقق والملفات ← تشخيص OOM والأداء.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["CPU vs GPU", "Parallelism", "VRAM & batch size", "Colab runtime", "Detect GPU", "OOM & performance", "Post-test"], active=0)
    h2("التوازي في أربع لقطات", "Parallelism in four snapshots")
    shown = [0, 1, 8, 16]
    caps = ["64 خلية مستقلة في ناتج ضرب مصفوفتين.", "بعد دفعة واحدة: CPU (4 أنوية) أنهى 4، وGPU أنهى 64.", "CPU في منتصف الطريق.", "CPU ينتهي بعد 16 دفعة. الأسبوع كله يشرح ما يترتب على هذا الفرق."]
    animation_player("w13_teaser", [Frame(parallel_svg(t), caption(c), action=f"tick {t}") for t, c in zip(shown, caps)], title_ar="نفس العمل، سرعتان", interval_ms=1800)
    table(["الموضوع", "درس الأسبوع", "الأسس التي يعتمد عليها"],
          [("لماذا GPU أسرع، VRAM، حجم الدفعة", "CPU/GPU والذاكرة", "الأسبوع 04: ضرب المصفوفات؛ الأسبوع 05: الدفعة وη"), ("Colab: التفعيل والتحقق والملفات والحفظ", "Colab", "الوحدة 22: البيئات"), ("OOM والأداء", "Colab", "الأسبوع 09: أشكال CNN")],
          ["rtl", "rtl", "rtl"])
    st.markdown(
        "**في نهاية الأسبوع ستستطيع:**\n"
        "- أن تشرح لماذا يسرّع GPU ضرب المصفوفات والالتفاف.\n"
        "- أن تقدّر ذاكرة خطوة تدريب وتختار دفعة تتسع.\n"
        "- أن تشغّل مشروعك في Colab على GPU وتتحقق منه وتحفظ عملك.\n"
        "- أن تقرأ رسالة OOM وتعالجها بالترتيب الصحيح."
    )
    practical_note("كل ما في المقرر عمل على CPU عمدًا ببيانات صغيرة. هذا الأسبوع يجهزك لمشروع أكبر دون تغيير في المنهج.")
    with st.container(horizontal=True):
        st.button("معمل GPU وذاكرة الدفعة", icon=":material/science:", on_click=go, args=("labs.gpu_batch_memory_lab",), key="w13_go_lab")
    intuition("GPU لا يغيّر الرياضيات ولا الدقة؛ يغيّر الزمن فقط. إن تغيّرت نتائجك كثيرًا عند الانتقال إلى GPU فقد تغيّر شيء آخر (الدفعة، η، البذرة).")
    h2("الاختبار القبلي", "Pre-test")
    quiz("course.w13.pretest", [
        Q("لماذا يسرّع GPU ضرب المصفوفات؟", ["أنويته أسرع", "آلاف الأنوية تحسب خلايا مستقلة معًا", "ذاكرته أكبر"], 1, ""),
        Q("الجزء من ذاكرة التدريب الذي ينمو مع الدفعة:", ["المعلمات", "حالة Adam", "التنشيطات"], 2, ""),
        Q("OOM: أول مقبض", ["η", "batch_size ÷ 2", "الحقب"], 1, ""),
        Q("في Colab، الملفات المرفوعة للجلسة…", ["تبقى دائمًا", "تُحذف عند انتهاء الجلسة", "تُرفع إلى Drive تلقائيًا"], 1, ""),
    ], title_ar="الاختبار القبلي — الأسبوع 13")
    takeaway("الأسبوع 13 = التوازي يفسّر السرعة، الذاكرة تحدد الدفعة، وColab يعطيك GPU إن أحسنت إدارة الجلسة.")
    lesson_footer(LESSON, ["التوازي في أربع لقطات (تحريك).", "خريطة الأسبوع.", "الاختبار القبلي."])
