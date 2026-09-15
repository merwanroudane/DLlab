import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.diagram import diagram, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto

LESSON = Lesson(
    id="course.w13.cpu_gpu_memory",
    title_ar="CPU مقابل GPU: حدس التوازي، ذاكرة VRAM، وحجم الدفعة مقابل الذاكرة",
    title_en="CPU vs GPU: Parallelism Intuition, VRAM & Batch Size vs Memory",
    module="course.w13",
    order=2,
    prerequisites=["course.w13.overview", "foundations.linalg.matrix_multiplication", "foundations.batch_epoch.batch_size_effects"],
    objectives_ar=["لماذا GPU أسرع: ضرب المصفوفات متوازٍ بإحراج؛ قياس نمو الزمن مع الحجم على CPU.", "ما الذي يشغل VRAM: معلمات، تدرجات، حالة المحسّن، وتنشيطات تتناسب مع الدفعة.", "قاعدة اختيار الدفعة: أكبر ما يتسع مع تعديل η، وmixed precision كأداة."],
    terms=["batch_size", "matrix_multiplication", "parameter"],
    labs=["labs.gpu_batch_memory_lab"],
    difficulty="intermediate",
    summary_ar="GPU = آلاف أنوية بسيطة لعمليات مستقلة (ضرب المصفوفات). VRAM محدودة: params×(4+4+8) + activations×batch×bytes. OOM → قلّل الدفعة أو float16. الدفعة الأكبر تحتاج η أكبر.",
)

CODE = '''import time, numpy as np
# 1) ضرب المصفوفات: العملية التي تشكّل معظم حساب الشبكة
for n in (256, 512, 1024, 2048):
    a = np.random.default_rng(0).normal(size=(n, n)).astype("float32")
    t = time.perf_counter(); c = a @ a; dt = time.perf_counter() - t
    print(f"n={n:<5} ops≈{2 * n**3 / 1e9:6.2f} GFLOP  time={dt * 1000:8.1f} ms  ->  {2 * n**3 / dt / 1e9:6.1f} GFLOP/s on this CPU")
print("doubling n multiplies work by 8 (cubic); a modern GPU reaches thousands of GFLOP/s on the same op")

# 2) أين تذهب ذاكرة التدريب؟ MLP مثال: 784 -> 512 -> 256 -> 10 (MNIST)
params = 784 * 512 + 512 + 512 * 256 + 256 + 256 * 10 + 10
acts_per_sample = 784 + 512 + 256 + 10                     # قيم التنشيط لملاحظة واحدة (تُحفظ للخلفي)
for batch in (32, 256, 2048):
    fixed = params * (4 + 4 + 8)                            # float32: المعلمات + التدرجات + Adam (m, v)
    acts = acts_per_sample * 4 * batch * 2                  # التنشيطات وتدرجاتها
    print(f"batch={batch:<5} params={params:,} fixed={fixed / 1e6:6.1f} MB  activations={acts / 1e6:6.1f} MB  total={(fixed + acts) / 1e6:6.1f} MB")
print("fixed part does not depend on batch; activations grow linearly with batch (and with image size for CNNs)")'''


def _svg() -> str:
    s = '<svg viewBox="0 0 720 200" width="100%" style="max-width:720px">' + svg_defs()
    s += svg_text(160, 25, "CPU: few powerful cores (4–16)", size=13, bold=True) + svg_text(540, 25, "GPU: thousands of simple cores", size=13, bold=True)
    for i in range(8):
        s += svg_box(40 + (i % 4) * 65, 45 + (i // 4) * 60, 55, 50, f"core {i}", "#E6F1FB", stroke="#2F6FB5", font=10)
    for i in range(20):
        for j in range(12):
            s += f'<rect x="{400 + j * 24}" y="{45 + i * 6}" width="20" height="4" fill="#E3F3F0" stroke="#1F7A78" stroke-width="0.5"/>'
    s += svg_text(160, 175, "sequential logic, branching, latency", size=11, color="#6B675F") + svg_text(540, 175, "same op on many numbers at once: matmul, conv", size=11, color="#6B675F")
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    definition("**CPU**: أنوية قليلة قوية مصممة للمنطق المتسلسل. **GPU**: آلاف الأنوية البسيطة تنفذ **نفس العملية على أرقام كثيرة معًا** (SIMD) — وهذا بالضبط ما يحتاجه ضرب المصفوفات والالتفاف. **VRAM**: ذاكرة البطاقة (4–80 GB) التي يجب أن تتسع للمعلمات والتدرجات وحالة المحسّن وتنشيطات الدفعة كاملة.")
    diagram("CPU مقابل GPU", _svg(), what_ar="يسار: 8 أنوية كبيرة. يمين: شبكة من أنوية صغيرة كثيرة.", how_ar="كل عنصر في ناتج ضرب مصفوفتين مستقل عن الآخر → يمكن حسابه بالتوازي. الشبكة العصبية = مصفوفات ضخمة → GPU.", takeaway_ar="GPU لا يفكر أسرع؛ يحسب أشياء كثيرة متشابهة معًا.", title_en="CPU vs GPU")
    why("لماذا هذا الأسبوع الآن؟ لأن كل ما سبق عمل على CPU في ثوانٍ بفضل بيانات صغيرة عمدًا. مشروعك (الأسبوع 14) قد يحتاج صورًا أكبر أو تسلسلات أطول — وعندها الفرق بين ساعات ودقائق هو GPU، والفرق بين «يعمل» و«OOM» هو فهم الذاكرة.")
    code_lab(CodeLab(
        key="w13_gpu", title_ar="نمو زمن ضرب المصفوفات على CPU، وأين تذهب ذاكرة التدريب", code=CODE, level="A",
        before=Before(goal_ar="قياس GFLOP/s لهذا الـ CPU على ضرب مصفوفات بأحجام متزايدة، ثم تفكيك ذاكرة خطوة تدريب MLP إلى جزء ثابت وجزء يتناسب مع الدفعة.", stage_ar="الأسبوع 13: التوازي والذاكرة.",
                      inputs_ar="مصفوفات عشوائية؛ MLP بأشكال MNIST.", expected_ar="الزمن ينمو ≈ ×8 مع مضاعفة n؛ عشرات GFLOP/s؛ الجزء الثابت ≈ 8.7 MB والتنشيطات تنمو خطيًا مع الدفعة."),
        explain=[("2-7", "GFLOP = 2n³ عمليات تقريبًا؛ الزمن على CPU يعطي عشرات GFLOP/s. GPU حديث: آلاف. الفرق ×50–×100 على نفس العملية."), ("10-16", "الذاكرة: 16 بايت لكل معلمة مع Adam (4 معلمة + 4 تدرج + 8 حالة)، ثم التنشيطات لكل ملاحظة × الدفعة × 2 (تُحفظ للخلفي، الأسس 14).")],
        run=run_printed(CODE),
        after_ar="- CNN على 224×224: التنشيطات أكبر بمئات المرات (H×W×C لكل طبقة) — لذلك OOM شائع في الصور لا في الجداول.\n- الجزء الثابت لا ينقذك عند OOM؛ الدفعة هي المقبض الأول.\n- **معمل GPU وذاكرة الدفعة** يحسب ذلك لأي CNN تختارها.",
    ))
    st.button("افتح معمل GPU وذاكرة الدفعة", icon=":material/science:", type="primary", on_click=goto, args=("labs.gpu_batch_memory_lab",), key="w13_lab_mem")
    equation(r"\text{VRAM} \approx \underbrace{P\,(4 + 4 + 8)}_{\text{params, grads, Adam}} \;+\; \underbrace{B \cdot A \cdot b \cdot 2}_{\text{activations}} \;+\; \text{overhead}",
             [("P", "عدد المعلمات."), ("B", "حجم الدفعة."), ("A", "عدد قيم التنشيط لملاحظة واحدة عبر كل الطبقات."), ("b", "بايت لكل قيمة: 4 (float32) أو 2 (float16).")],
             meaning_ar="الحد الأول ثابت؛ الثاني يتناسب خطيًا مع الدفعة وتربيعيًا مع دقة الصورة.", example_ar="ResNet-50 (25M معلمة): الجزء الثابت ≈ 400 MB؛ التنشيطات لدفعة 64 صورة 224×224 عدة GB.", dl_link_ar="`ResourceExhaustedError: OOM when allocating tensor with shape[64, 56, 56, 256]` — الشكل يخبرك أي تنشيط تجاوز الذاكرة.", title_ar="تقدير الذاكرة")
    h2("حجم الدفعة: الذاكرة والسرعة والتعلم", "Batch size: memory, speed, learning")
    compare_table(["الدفعة", "الذاكرة", "السرعة/حقبة", "التدرج", "η", "ملاحظة"],
                  [("صغيرة (8–32)", "قليلة", "أبطأ (تحديثات أكثر، استغلال أقل لـ GPU)", "ضوضائي (تنظيم ضمني)", "أصغر", "جيدة للبيانات الصغيرة"), ("متوسطة (64–256)", "معتدلة", "جيدة", "مستقر", "متوسط", "الافتراضي العملي على GPU"), ("كبيرة (512+)", "كبيرة", "أسرع/حقبة لكن حقب أكثر أحيانًا", "دقيق جدًا (قد يعمّم أسوأ)", "أكبر (قاعدة الجذر/الخطي)", "تحتاج ضبط η وإحماء")],
                  ["rtl", "rtl", "rtl", "rtl", "rtl", "rtl"])
    intuition("اختر الدفعة بالذاكرة أولًا (أكبر ما يتسع مع هامش)، ثم اضبط η لها (الأسبوع 05: الدفعة الأكبر تتحمل η أكبر). لا تضاعف الدفعة وتبقي η — ستتباطأ. و«mixed precision» (float16 للتنشيطات) يوفر نصف الذاكرة ويسرّع على GPU حديثة.")
    research_note("**تعميق**: `keras.mixed_precision.set_global_policy('mixed_float16')` يجعل الحسابات float16 والأوزان float32. يقلل الذاكرة ويسرّع على بطاقات بنوى Tensor؛ قد يحتاج `LossScaleOptimizer` (تلقائي في Keras) لتفادي تلاشي التدرجات الصغيرة في float16.")
    common_mistake("«GPU أعطى دقة أعلى». GPU لا يغيّر الرياضيات؛ الفروق الصغيرة في الكسور العشرية من ترتيب الجمع غير الحتمي. إن اختلفت الدقة كثيرًا فقد غيّرت الدفعة أو η معه دون قصد.")
    quiz("w13.gpu", [
        Q("مضاعفة n في ضرب مصفوفتين n×n تضاعف العمل…", ["×2", "×4", "×8"], 2, "تكعيبي."),
        Q("الجزء من ذاكرة التدريب الذي ينمو مع الدفعة:", ["المعلمات", "حالة Adam", "التنشيطات"], 2, ""),
        Q("OOM: أول مقبض", ["η", "batch_size ÷ 2", "الحقب"], 1, ""),
        Q("مضاعفة الدفعة مع إبقاء η…", ["تسرّع التعلم", "تبطئه غالبًا: عدّل η", "لا تؤثر"], 1, ""),
    ])
    takeaway("GPU = توازٍ لعمليات متشابهة (matmul/conv) ×50–100. VRAM = ثابت (P×16 B) + تنشيطات (B×A×b×2). OOM → الدفعة ÷ 2، float16، شبكة/صورة أصغر. الدفعة الأكبر تحتاج η أكبر.")
    lesson_footer(LESSON, ["التوازي بالقياس.", "معادلة الذاكرة.", "الدفعة: ذاكرة وسرعة وη."])
