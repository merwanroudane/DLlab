import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.diagram import diagram, svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.frameworks.keras",
    title_ar="ما هي Keras؟ خريطة سير العمل",
    title_en="What is Keras? The Keras Workflow Map",
    module="foundations.frameworks",
    order=3,
    prerequisites=["foundations.frameworks.ecosystem_map"],
    objectives_ar=["فهم Keras كواجهة عليا للتعلم العميق ولماذا صُممت لتقليل الكود المتكرر.", "خريطة سير العمل: Layers → Model → compile → fit → evaluate/predict → save.", "معنى الخلفية ببساطة، والتصريح بأن المقرر يستخدم Keras 3 فوق TensorFlow."],
    terms=["model", "loss", "optimizer", "epoch", "batch_size"],
    difficulty="beginner",
    summary_ar="Keras: تصف النموذج كطبقات، تحدد الخسارة والمحسّن والمقاييس في compile، وتدرّب بـ fit. الصفحات الفرعية تفصّل كل مرحلة وكل مخرج.",
)

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, keras
from keras import layers
keras.utils.set_random_seed(0)

# بيانات صغيرة: y = 3x1 - 2x2 + 1 + ضوضاء  (انحدار)
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 2)).astype("float32")
y = (3 * X[:, 0] - 2 * X[:, 1] + 1 + 0.1 * rng.normal(size=200)).astype("float32")

model = keras.Sequential([layers.Input(shape=(2,)), layers.Dense(1)])   # 1) الطبقات → النموذج
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1), loss="mse")  # 2) الخسارة والمحسّن
history = model.fit(X, y, epochs=20, batch_size=20, verbose=0)         # 3) حلقة التدريب مخفية
W, b = model.layers[0].get_weights()
print("backend        :", keras.backend.backend())
print("learned W      :", W.ravel().round(3), "(true: [3, -2])")
print("learned b      :", b.round(3), "(true: 1)")
print("loss per epoch :", np.round(history.history["loss"][::5], 4), "... final", round(history.history["loss"][-1], 5))
print("predict([1, 1]):", model.predict(np.array([[1.0, 1.0]], "float32"), verbose=0).ravel().round(3), "(true: 2)")'''


def _workflow_svg() -> str:
    steps = [("Layers", "#E6F1FB"), ("Model", "#E3F3F0"), ("compile()", "#FBE6E2"), ("fit()", "#EFE9F8"), ("evaluate() / predict()", "#FFF3D6"), ("save()", "#F1EFEA")]
    s = '<svg viewBox="0 0 760 150" width="100%" style="max-width:760px">' + svg_defs()
    x = 10
    widths = [90, 90, 110, 90, 190, 90]
    for (lbl, fill), w in zip(steps, widths):
        s += svg_box(x, 40, w, 50, lbl, fill, font=13, bold=True)
        x += w + 14
        if lbl != "save()":
            s += svg_arrow(x - 13, 65, x - 1, 65)
    subs = ["Dense, Conv2D, LSTM…", "Sequential / Functional", "optimizer + loss + metrics", "epochs, batch_size, validation, callbacks → History", "test loss / metrics · predictions", ".keras file"]
    x = 10
    for sub, w in zip(subs, widths):
        s += svg_text(x + w / 2, 115, sub, size=10, color="#6B675F")
        x += w + 14
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("ما هي Keras؟", "What is Keras?")
    definition("**Keras**: واجهة برمجية عليا للتعلم العميق `high-level deep-learning API`. تصف النموذج بطبقات، وتغلّف الخسارة والمحسّن والمقاييس في `compile()`، وحلقة التدريب كاملة في `fit()`. في هذا المقرر تعمل **Keras 3 بخلفية TensorFlow** — أي أن TensorFlow هو من ينفّذ ضرب المصفوفات والاشتقاق والأجهزة تحتها.")
    why("صُممت Keras لتقليل **الكود المتكرر** (boilerplate): حلقة الدفعات والحقب، حساب المقاييس، التحقق، الحفظ، شريط التقدم — كلها متطابقة تقريبًا في 95% من المشاريع. بدل إعادة كتابتها (وأخطائها) في كل مرة، تصفها بمعاملات: `epochs=`, `batch_size=`, `validation_data=`, `callbacks=`. ما تخسره: الرؤية الصريحة لكل خطوة — ولهذا بنيت الحلقة يدويًا في الوحدة 16 قبل أن تُخفيها.")
    diagram("خريطة سير عمل Keras", _workflow_svg(), what_ar="ست مراحل متتابعة. كل مرحلة كائن أو استدعاء واحد، ولكل واحدة صفحة فرعية في هذه الوحدة.",
            how_ar="طبقات تُركَّب في نموذج؛ compile يربط النموذج بالخسارة والمحسّن والمقاييس (لا يدرّب شيئًا)؛ fit يشغّل حلقة التدريب ويعيد History؛ evaluate يقيس على بيانات لم تُدرَّب؛ predict يعطي المخرجات؛ save يكتب ملفًا.",
            takeaway_ar="كل مرحلة تقابل شيئًا بنيته يدويًا: الطبقات (وحدة 10–12)، compile (13، 15)، fit (16–17)، evaluate (18).", title_en="Keras workflow")
    h2("المكوّنات الستة", "The six building blocks")
    compare_table(["المكوّن", "ما هو", "بنيته يدويًا في", "في Keras"],
                  [("Layer", "تحويل بمعلمات أو بدونها: كثيفة، تنشيط، Dropout، BN، Conv…", "وحدات 10–12، 20", "`keras.layers.Dense(64, activation='relu')`"),
                   ("Model", "سلسلة/رسم من الطبقات مع forward محدد", "وحدة 12 (`forward`)", "`keras.Sequential([...])` أو Functional API"),
                   ("Loss", "الدالة التي نقلّلها", "وحدة 13", "`loss='mse'` أو `keras.losses.BinaryCrossentropy()`"),
                   ("Metric", "ما نراقبه (لا يُشتق)", "وحدة 18", "`metrics=['accuracy']`"),
                   ("Optimizer", "قاعدة التحديث", "وحدة 15", "`keras.optimizers.Adam(learning_rate=1e-3)`"),
                   ("Callback", "كود يُستدعى في أحداث الحلقة (نهاية حقبة…)", "وحدة 16 (الإيقاف المبكر يدويًا)", "`keras.callbacks.EarlyStopping(...)`")],
                  ["ltr", "rtl", "rtl", "code"])
    code_lab(CodeLab(
        key="keras_first", title_ar="أول نموذج Keras: انحدار خطي في 3 أسطر جوهرية", code=CODE, level="B",
        before=Before(goal_ar="بناء وتدريب نموذج Keras على نفس نوع المسألة التي حللتها يدويًا في الوحدة 16، والتحقق من أنه تعلّم المعاملات الحقيقية.", stage_ar="Keras: الطبقات → النموذج → compile → fit → predict.",
                      inputs_ar="X بشكل (200, 2)، y بشكل (200,) من علاقة خطية معلومة.", expected_ar="اسم الخلفية (tensorflow)، أوزان قريبة من [3, −2] وانحياز قريب من 1، خسارة تنخفض، تنبؤ قريب من 2.",
                      objects_ar="`keras.Sequential`، `layers.Dense`، `keras.optimizers.SGD`، `History`."),
        explain=[("1-4", "إسكات سجلات TensorFlow، استيراد keras وطبقاته، تثبيت البذرة (وحدة 2: الاستنساخ)."), ("7-9", "بيانات بعلاقة معلومة كي نتحقق من الأوزان المتعلَّمة."),
                 ("11", "`Input(shape=(2,))` يصف شكل ملاحظة واحدة (بدون بُعد الدفعة). `Dense(1)` = طبقة خطية z = xW + b بلا تنشيط: انحدار."), ("12", "`compile` يربط المحسّن والخسارة بالنموذج. لا حساب هنا."),
                 ("13", "`fit` = حلقة الوحدة 16 كاملة: 200/20 = 10 دفعات × 20 حقبة = 200 تحديث."), ("14-19", "استخراج الأوزان ومقارنتها بالحقيقة، وقراءة `history.history`.")],
        run=run_printed(CODE),
        after_ar="- `backend: tensorflow` — التصريح الذي يجب أن تعرفه: Keras هنا تنفذ عبر TensorFlow.\n- الأوزان ≈ [3, −2] وb ≈ 1: نفس ما يعطيه الانحدار اليدوي في الوحدة 16.\n- `history.history['loss']` قائمة بطول 20 = حقبة لكل عنصر.",
    ))
    intuition("قارن مع كود الوحدة 16: هناك كتبت الحلقة والاشتقاق والتحديث (~15 سطرًا). هنا 3 أسطر جوهرية. الفرق ليس في الرياضيات — بل في **من يكتب الحلقة**.")
    h2("الخلفية ببساطة", "The backend, simply")
    st.markdown("""
Keras لا تحسب شيئًا بنفسها. عندما تكتب `layers.Dense(64)` فإنها تسجّل «طبقة كثيفة بـ 64 وحدة»؛ عند التنفيذ تطلب من **الخلفية** ضرب المصفوفات والاشتقاق والذاكرة والأجهزة. الخلفية في هذا المقرر هي **TensorFlow** — لذلك عندما تظهر رسالة خطأ تحمل `tf.` أو `tensorflow`، فهي من الطبقة التي تحت Keras.
""")
    research_note("**تعميق (غير أساسي)**: Keras 3 يمكنها العمل بخلفية JAX أو PyTorch أيضًا بضبط `KERAS_BACKEND` قبل الاستيراد. نفس كود النموذج يعمل، لكن الموترات المرجعة تكون من نوع الخلفية المختارة. لا نستخدم ذلك في المقرر؛ نذكره حتى لا تربكك مقالات تتحدث عن «Keras بدون TensorFlow».")
    h2("خريطة الصفحات الفرعية", "Sub-page map")
    st.markdown("""
1. **الطبقات والنماذج** — Layers، Models، Sequential، Functional، Subclassing (تعميق).
2. **compile()** — المحسّن، الخسارة، المقاييس.
3. **fit()** — epochs، batch_size، validation_split/data، shuffle، callbacks، verbose، History + محرك fit المرئي.
4. **evaluate() و predict()**.
5. **الاستدعاءات والحفظ** — EarlyStopping، ModelCheckpoint، جداول معدل التعلم، استدعاءات مخصصة، save/load.
6. **تفكيك model.summary()** وعدّ المعلمات.
7. **مستكشف المخرجات** — كل مخرج مع «كيف أقرأه؟».
8. **أخطاء Keras الشائعة** — الأشكال، الخسارة/التنشيط، ترميز الهدف، سلوك GPU.
9. **معيار معمل كود Keras** — مثال كامل بكل بنود العقد.
""")
    common_mistake("«compile يدرّب النموذج». لا: `compile` يربط فقط (المحسّن، الخسارة، المقاييس). التدريب يبدأ في `fit`. ولذلك يمكن إعادة `compile` بمحسّن آخر دون فقدان الأوزان.")
    quiz("keras.intro", [
        Q("Keras في هذا المقرر تعمل بخلفية…", ["NumPy", "TensorFlow", "PyTorch"], 1, "مُصرَّح به."),
        Q("`compile()`…", ["يدرّب", "يربط النموذج بالخسارة والمحسّن والمقاييس", "يحفظ"], 1, "لا حساب."),
        Q("`fit()` يقابل يدويًا…", ["التمرير الأمامي فقط", "حلقة التدريب كاملة (وحدة 16)", "الخسارة فقط"], 1, "الحلقة."),
        Q("ما تخسره باستخدام fit بدل حلقة صريحة؟", ["الدقة", "الرؤية الصريحة لكل خطوة", "السرعة"], 1, "التحكم/الشفافية."),
    ])
    takeaway("Keras = واجهة عليا فوق TensorFlow (هنا). ست مراحل: Layers → Model → compile → fit → evaluate/predict → save. لا سحر: كل مرحلة بنيتها يدويًا من قبل.")
    lesson_footer(LESSON, ["تعريف Keras والخلفية.", "الخريطة والمكوّنات الستة.", "أول نموذج في 3 أسطر."])
