import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.diagram import diagram, svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.frameworks.tensorflow",
    title_ar="ما هو TensorFlow؟ المنظومة تحت Keras",
    title_en="What is TensorFlow? The System Beneath Keras",
    module="foundations.frameworks",
    order=13,
    prerequisites=["foundations.frameworks.keras.code_lab_standard"],
    objectives_ar=["فهم TensorFlow كإطار أوسع من Keras: موترات، عمليات، اشتقاق تلقائي، تنفيذ فوري/رسمي، أجهزة، tf.data، حفظ، TensorBoard.", "معرفة متى تنزل من Keras إلى TensorFlow ولماذا.", "خريطة الصفحات الفرعية."],
    terms=["tensor", "shape", "dtype"],
    difficulty="intermediate",
    summary_ar="TensorFlow = الطبقة التي تنفذ ما تطلبه Keras: tf.Tensor وعملياته، tf.Variable، GradientTape للاشتقاق، tf.data لخطوط البيانات، الأجهزة، والحفظ. Keras تستدعيه؛ أنت تنزل إليه عند الحاجة إلى تحكم أدق.",
)

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf
print("TensorFlow", tf.__version__, "| eager execution:", tf.executing_eagerly())

# 1) موتر: مصفوفة + dtype + جهاز
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print("x =", x.numpy().tolist(), "| shape", x.shape, "| dtype", x.dtype.name, "| device", x.device.split("/")[-1])

# 2) عمليات: نفس رياضيات NumPy
W = tf.constant([[0.5], [-1.0]]); b = tf.constant([0.1])
z = tf.matmul(x, W) + b                       # (2,2)@(2,1)+(1,) -> (2,1) بالبث
print("z = x@W + b =", z.numpy().ravel().round(3).tolist(), "| tf.nn.sigmoid(z) =", tf.nn.sigmoid(z).numpy().ravel().round(3).tolist())

# 3) متغير: موتر قابل للتحديث (المعلمات)
w = tf.Variable(2.0)
# 4) اشتقاق تلقائي: يسجل العمليات ويحسب المشتقة
with tf.GradientTape() as tape:
    loss = (w * 3.0 - 1.0) ** 2               # L = (3w - 1)^2  ->  dL/dw = 2(3w-1)*3 = 6(3w-1) = 30 عند w=2
g = tape.gradient(loss, w)
print("loss =", float(loss), "| dL/dw =", float(g), "(hand: 30.0)")
w.assign_sub(0.01 * g)                        # خطوة انحدار تدرج يدوية
print("w after one SGD step:", round(float(w), 4))

# 5) تبادل مع NumPy بلا نسخ مفاهيمي
print("numpy -> tensor:", tf.convert_to_tensor(np.arange(3.0)).dtype.name, "| tensor -> numpy:", type(z.numpy()).__name__)'''


def _stack_svg() -> str:
    s = '<svg viewBox="0 0 760 250" width="100%" style="max-width:760px">' + svg_defs()
    s += svg_box(230, 15, 300, 44, "Keras  (layers · compile · fit · callbacks)", "#E3F3F0", stroke="#1F7A78", font=13, bold=True)
    s += svg_arrow(380, 61, 380, 78)
    s += f'<rect x="20" y="80" width="720" height="150" rx="12" fill="#FBE6E2" stroke="#C8473A"/>'
    s += svg_text(380, 100, "TensorFlow", size=15, bold=True, color="#C8473A")
    items = ["tf.Tensor / ops", "tf.Variable", "GradientTape", "tf.data", "devices CPU/GPU", "saving / export", "tf.function (graph)", "TensorBoard logs"]
    for i, it in enumerate(items):
        col, row = i % 4, i // 4
        s += svg_box(35 + col * 178, 115 + row * 52, 165, 40, it, "#FFFDF9", stroke="#B9B2A6", font=12)
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    definition("**TensorFlow**: إطار تعلم عميق يوفّر (1) **الموترات** `tf.Tensor` وعملياتها على CPU/GPU، (2) **المتغيرات** `tf.Variable` للمعلمات، (3) **الاشتقاق التلقائي** عبر `tf.GradientTape`، (4) **خطوط البيانات** `tf.data`، (5) الحفظ والتصدير، (6) أدوات المراقبة (TensorBoard). Keras تبني فوق هذه الستة.")
    diagram("Keras فوق TensorFlow", _stack_svg(), what_ar="كل استدعاء في Keras (fit، Dense، Adam) يتحول إلى عمليات على هذه الصناديق الثمانية.", how_ar="عندما تكفيك Keras تبقى في الصندوق الأخضر. تنزل إلى الأحمر عندما تحتاج: تدرجات مخصصة، حلقة تدريب غير قياسية، خط بيانات معقد، أو تشخيص دقيق.",
            takeaway_ar="Keras ≠ TensorFlow: الأولى واجهة، والثاني المحرك. رسائل الخطأ التي تحمل tf. قادمة من الأسفل.", title_en="Keras on TensorFlow")
    why("لماذا تتعلم الطبقة السفلى إن كانت Keras تكفي؟ ثلاثة أسباب: (1) **التشخيص**: أخطاء الأشكال وdtype والأجهزة تأتي من TensorFlow وبلغته. (2) **التحكم**: بعض الأبحاث تحتاج حلقة تدريب مخصصة أو خسارة بتدرج خاص. (3) **الفهم**: `GradientTape` هو الوحدة 14 حيًّا — تراه يحسب التدرج الذي حسبته بيدك.")
    h2("المفاهيم الستة في مثال واحد", "The six concepts in one example")
    code_lab(CodeLab(
        key="tf_intro", title_ar="موتر، عملية، متغير، شريط تدرج، خطوة تحديث", code=CODE, level="B",
        before=Before(goal_ar="لمس كل مفهوم أساسي في TensorFlow في 20 سطرًا، والتحقق من أن التدرج التلقائي يطابق الاشتقاق اليدوي.", stage_ar="TensorFlow: الأساسيات.",
                      inputs_ar="مصفوفة 2×2 ومتغير عددي.", expected_ar="shape (2, 2) وdtype float32 وجهاز CPU؛ z بالبث؛ dL/dw = 30 بالضبط؛ w ينخفض بعد الخطوة.",
                      objects_ar="`tf.constant`, `tf.matmul`, `tf.Variable`, `tf.GradientTape`."),
        explain=[("2-3", "`executing_eagerly() = True`: TensorFlow 2 ينفذ كل سطر فورًا (كـ NumPy) — لا رسم مؤجل إلا إن طلبته بـ `tf.function`."), ("6-7", "الموتر = قيم + شكل + dtype + جهاز. `float32` الافتراضي للكسور (وحدة 1: dtype)."),
                 ("10-12", "matmul والبث كما في NumPy (وحدة 4). `tf.nn` تحوي التنشيطات."), ("15", "`tf.Variable`: موتر **قابل للتغيير في مكانه** — هذا ما تكون عليه أوزان الطبقات."),
                 ("17-21", "الشريط يسجل العمليات على المتغيرات داخل `with`؛ `gradient(loss, w)` يطبق قاعدة السلسلة (وحدة 14). النتيجة 30 = ما تعطيه اليد."), ("22-23", "`assign_sub` = θ ← θ − η·g: خطوة المحسّن (وحدة 15) مكتوبة يدويًا."), ("26", "التبادل مع NumPy مجاني على CPU.")],
        run=run_printed(CODE),
        after_ar="- `device: CPU:0` على هذه الآلة؛ مع GPU تُنشأ الموترات عليه تلقائيًا.\n- التدرج 30 يطابق اليد: هذا هو «الاشتقاق التلقائي» — لا تقريب عددي بل قاعدة سلسلة رمزية على العمليات المسجلة.\n- Keras `fit` تفعل الأسطر 17–23 لكل دفعة ولكل معلمة.",
    ))
    h2("التنفيذ الفوري مقابل الرسم", "Eager vs graph execution")
    compare_table(["", "فوري (Eager)", "رسم (Graph / tf.function)"],
                  [("متى", "افتراضي في TF2؛ كل سطر ينفذ فورًا", "عند تزيين دالة بـ `@tf.function` أو داخل Keras fit تلقائيًا"), ("المزايا", "تصحيح سهل: print، pdb، أشكال فورية", "أسرع (تحسين ودمج العمليات)، قابل للتصدير"),
                   ("العيوب", "أبطأ قليلًا", "الأخطاء تظهر عند التتبع؛ print يُنفَّذ مرة واحدة؛ قيود على بايثون الحر"), ("في المقرر", "كل الأمثلة", "تعميق فقط")],
                  ["rtl", "rtl", "rtl"])
    research_note("**تعميق**: `tf.function` تتبّع الدالة مرة (trace) لتبني رسمًا حسابيًا ثم تنفذه مجمَّعًا. لهذا `print` داخلها يعمل مرة واحدة، ولهذا تكون حقبة Keras الأولى أبطأ. لا تحتاج هذا في المسار الأساسي؛ Keras تستخدمه لك.")
    h2("خريطة الصفحات الفرعية", "Sub-page map")
    st.markdown("""
1. **الموترات** — tf.Tensor، الإنشاء، الشكل، الرتبة، dtype، العمليات، البث، constant مقابل Variable، الأجهزة + **مستكشف موترات TensorFlow**.
2. **GradientTape** — من دالة بسيطة إلى شبكة: يدويًا مقابل تلقائيًا + المعمل.
3. **tf.data** — Dataset، batch، shuffle، prefetch.
4. **علاقة Keras بـ TensorFlow + حلقة تدريب مخصصة** (تعميق) + الحفظ/التصدير.
5. **TensorBoard وأخطاء TensorFlow الشائعة** — الأجهزة، dtype، الأشكال.
""")
    common_mistake("`import tensorflow.keras` مقابل `import keras`: في هذا المشروع Keras 3 حزمة مستقلة (`import keras`) تعمل بخلفية TensorFlow. `tf.keras` موجود للتوافق ويشير إلى Keras 3 أيضًا في TF 2.16+. استخدم `import keras` كما في كل الأمثلة.")
    quiz("tf.intro", [
        Q("`tf.Variable` مقابل `tf.constant`:", ["لا فرق", "Variable قابل للتحديث في مكانه (المعلمات)", "constant أسرع"], 1, "الأوزان Variables."),
        Q("`GradientTape` يحسب التدرج بـ…", ["الفروق المنتهية", "قاعدة السلسلة على العمليات المسجلة", "التخمين"], 1, "اشتقاق تلقائي."),
        Q("TF2 افتراضيًا…", ["يبني رسمًا مؤجلًا", "ينفذ فوريًا (eager)", "لا ينفذ"], 1, "eager."),
        Q("Keras بالنسبة لـ TensorFlow…", ["منافس", "واجهة عليا فوقه", "نفس الشيء"], 1, "واجهة."),
    ])
    takeaway("TensorFlow = موترات + متغيرات + GradientTape + tf.data + أجهزة + حفظ. فوري افتراضيًا. Keras تستدعيه؛ تنزل إليه للتشخيص والتحكم والفهم.")
    lesson_footer(LESSON, ["الصناديق الثمانية تحت Keras.", "مثال المفاهيم الستة.", "فوري مقابل رسم."])
