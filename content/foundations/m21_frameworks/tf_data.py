import streamlit as st

from components.callouts import common_mistake, definition, intuition, practical_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import pipeline

LESSON = Lesson(
    id="foundations.frameworks.tensorflow.tf_data",
    title_ar="tf.data.Dataset: الدفعات والخلط والجلب المسبق",
    title_en="tf.data.Dataset: Batching, Shuffling & Prefetch",
    module="foundations.frameworks",
    parent="foundations.frameworks.tensorflow",
    order=16,
    prerequisites=["foundations.frameworks.tensorflow.tensors", "foundations.prep.tensors_batching_pipelines", "foundations.batch_epoch.definitions"],
    objectives_ar=["بناء Dataset من مصفوفات، وتطبيق map/shuffle/batch/prefetch وفهم ترتيبها.", "معنى buffer_size في shuffle، وdrop_remainder في batch، وAUTOTUNE في prefetch.", "متى تكفي مصفوفات NumPy ومتى تحتاج tf.data."],
    terms=["batch", "batch_size", "epoch", "iteration"],
    labs=["labs.epoch_batch_simulator"],
    difficulty="intermediate",
    summary_ar="Dataset = تدفق كسول من الأمثلة. from_tensor_slices → map (تحويل) → shuffle(buffer) → batch(B) → prefetch(AUTOTUNE). fit يقبله مباشرة. الترتيب مهم: shuffle قبل batch.",
)

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf
X = np.arange(10, dtype="float32").reshape(10, 1) * 10   # 10 ملاحظات: 0, 10, ..., 90
y = np.arange(10)

ds = tf.data.Dataset.from_tensor_slices((X, y))           # كل عنصر = (ملاحظة واحدة، وسمها)
print("element_spec:", ds.element_spec)
print("cardinality  :", int(ds.cardinality()), "examples")
print("first 3      :", [(int(a[0]), int(b)) for a, b in ds.take(3)])

# 1) map: تحويل لكل مثال (مثلًا تحجيم) — كسول: لا ينفذ حتى نمرّ
scaled = ds.map(lambda a, b: (a / 100.0, b))
# 2) shuffle: buffer_size = كم مثالًا يحمّل قبل السحب العشوائي
# 3) batch: تجميع B أمثلة في موتر (B, ...)
# 4) prefetch: تجهيز الدفعة التالية أثناء تدريب الحالية
pipe = scaled.shuffle(buffer_size=10, seed=0).batch(4).prefetch(tf.data.AUTOTUNE)
print("after batch, element_spec:", pipe.element_spec)
for epoch in range(2):
    print(f"epoch {epoch+1}:", [b.numpy().tolist() for _, b in pipe])   # ترتيب مختلف كل حقبة، آخر دفعة أصغر (10 = 4+4+2)
print("drop_remainder=True ->", [b.numpy().tolist() for _, b in scaled.batch(4, drop_remainder=True)])

# الترتيب مهم: batch ثم shuffle يخلط الدفعات لا الأمثلة
wrong = scaled.batch(4).shuffle(10, seed=0)
print("batch->shuffle (wrong):", [b.numpy().tolist() for _, b in wrong], "<- batches keep their members")

# buffer_size صغير = خلط ضعيف
weak = scaled.shuffle(buffer_size=2, seed=0).batch(10)
print("shuffle(buffer=2) :", [b.numpy().tolist() for _, b in weak][0], "<- nearly sorted")
full = scaled.shuffle(buffer_size=10, seed=0).batch(10)
print("shuffle(buffer=10):", [b.numpy().tolist() for _, b in full][0], "<- true permutation")

# fit يقبل Dataset مباشرة (بلا batch_size لأنه داخل الخط)
import keras
m = keras.Sequential([keras.layers.Input(shape=(1,)), keras.layers.Dense(1)]); m.compile(optimizer="sgd", loss="mse")
h = m.fit(scaled.map(lambda a, b: (a, tf.cast(b, tf.float32))).shuffle(10).batch(5), epochs=2, shuffle=False, verbose=0)   # الخلط داخل الخط
print("fit on Dataset -> epochs:", len(h.history["loss"]), "| steps/epoch = 10/5 = 2")'''


def render() -> None:
    lesson_header(LESSON)
    definition("**`tf.data.Dataset`**: تدفق **كسول** (lazy) من الأمثلة يُبنى بسلسلة تحويلات: `from_tensor_slices` (المصدر) → `map` (تحويل لكل مثال) → `shuffle` (خلط بمخزن) → `batch` (تجميع) → `prefetch` (تجهيز مسبق). لا يُحسب شيء حتى تمرّ عليه أو يمرّ عليه `fit`.")
    pipeline(["from_tensor_slices / files", "map(preprocess)", "shuffle(buffer)", "batch(B)", "prefetch(AUTOTUNE)", "model.fit(ds)"], active=3)
    why("مصفوفات NumPy كافية عندما تتسع البيانات في الذاكرة. مع صور كثيرة أو ملفات ضخمة تحتاج قراءة تدريجية، وتحويلًا متوازيًا، وتجهيزًا للدفعة التالية بينما تتدرب الحالية على GPU. `tf.data` يفعل ذلك؛ وفهم `shuffle` و`batch` هنا يفسّر ما كان `fit` يفعله بصمت مع المصفوفات (وحدة 17).")
    code_lab(CodeLab(
        key="tf_data", title_ar="خط بيانات كامل، وترتيب التحويلات، وحجم مخزن الخلط", code=CODE, level="B",
        before=Before(goal_ar="بناء Dataset من 10 ملاحظات وتتبع أثر كل تحويل على ما يخرج، وإظهار خطأين شائعين (batch قبل shuffle، buffer صغير)، ثم تمريره إلى fit.", stage_ar="TensorFlow: خط البيانات.",
                      inputs_ar="X بشكل (10, 1)، y = 0..9 (لتتبّع الترتيب).", expected_ar="element_spec قبل وبعد batch؛ ترتيب مختلف كل حقبة؛ آخر دفعة بحجم 2؛ batch→shuffle يبقي أعضاء الدفعات؛ buffer=2 شبه مرتب؛ fit يعمل بلا batch_size."),
        explain=[("6-9", "`from_tensor_slices` يقطع المحور 0: كل عنصر ملاحظة واحدة. `element_spec` يصف الشكل وdtype لعنصر واحد، `cardinality` عدد الأمثلة."),
                 ("12-17", "التحويلات الأربع بترتيبها الصحيح. `AUTOTUNE` يترك TensorFlow يقرر كم دفعة يجهّز مسبقًا."), ("18-20", "بعد batch يصبح الشكل `(None, 1)`: الدفعة الأخيرة أصغر ما لم تُسقطها بـ `drop_remainder`. كل مرور جديد = خلط جديد = حقبة."),
                 ("23-24", "**خطأ شائع 1**: batch ثم shuffle يخلط ترتيب الدفعات فقط؛ أعضاء كل دفعة ثابتون كل حقبة."), ("27-30", "**خطأ شائع 2**: `buffer_size` أصغر من البيانات = خلط ضعيف. للخلط الكامل buffer ≥ عدد الأمثلة (أو كبير جدًا للملفات الضخمة)."),
                 ("33-36", "`fit` يقبل Dataset مباشرة؛ `batch_size` يُحدد في الخط لا في fit.")],
        run=run_printed(CODE),
        after_ar="- ترتيب الحقبتين مختلف: هذا ما يفعله `shuffle=True` في fit مع المصفوفات.\n- `(None, 1)` بعد batch = نفس `None` في summary: حجم الدفعة متغير.\n- مع y أعداد صحيحة كان لزامًا `tf.cast` إلى float32 لخسارة mse — قاعدة dtype الصارمة من درس الموترات.",
    ))
    h2("المعاملات التي تهم", "Parameters that matter")
    compare_table(["التحويل", "المعامل", "المعنى", "القيمة العملية"],
                  [("`shuffle`", "`buffer_size`", "كم مثالًا يُحمَّل في المخزن ويُسحب منه عشوائيًا", "≥ عدد الأمثلة للخلط الكامل؛ 1000–10000 للملفات الضخمة"), ("`shuffle`", "`reshuffle_each_iteration`", "خلط جديد كل حقبة", "True (افتراضي)"),
                   ("`batch`", "`batch_size`", "حجم الدفعة", "كما في وحدة 17"), ("`batch`", "`drop_remainder`", "إسقاط الدفعة الأخيرة الناقصة", "True عند الحاجة لشكل ثابت (BN بدفعة صغيرة، TPU)"),
                   ("`map`", "`num_parallel_calls`", "تحويل متوازٍ", "`tf.data.AUTOTUNE`"), ("`prefetch`", "`buffer_size`", "كم دفعة تُجهَّز مسبقًا", "`tf.data.AUTOTUNE`"), ("`cache`", "—", "حفظ نتيجة map في الذاكرة/القرص بعد أول حقبة", "قبل shuffle، إن اتسعت الذاكرة")],
                  ["code", "code", "rtl", "rtl"])
    intuition("`shuffle(buffer_size=k)`: يملأ مخزنًا بأول k أمثلة، يسحب واحدًا عشوائيًا، يملأ مكانه بالمثال التالي… لذلك مع k صغير لا يمكن لمثال في آخر البيانات أن يظهر مبكرًا — خلط «محلي». مع k = n خلط كامل.")
    practical_note("الترتيب القياسي: `ds.map(...).cache().shuffle(n).batch(B).prefetch(AUTOTUNE)`. للتحقق/الاختبار: بلا shuffle (`ds_val.batch(B)`). ولا تخلط السلاسل الزمنية داخل النافذة (الأسبوع 10).")
    common_mistake("`shuffle` بعد `batch` — الخطأ الأكثر شيوعًا. النتيجة: كل حقبة نفس الدفعات بترتيب مختلف؛ تدرجات مترابطة وتعميم أضعف (وحدة 17: لماذا الخلط).")
    quiz("tf.data", [
        Q("الترتيب الصحيح:", ["batch → shuffle → map", "map → shuffle → batch → prefetch", "prefetch → batch → shuffle"], 1, "خلط الأمثلة قبل التجميع."),
        Q("`shuffle(buffer_size=2)` على 1000 مثال يعطي…", ["خلطًا كاملًا", "خلطًا محليًا ضعيفًا", "خطأ"], 1, "مخزن صغير."),
        Q("`fit(ds)` مع Dataset مقسّم بـ batch(32): batch_size في fit", ["يجب 32", "لا يُمرَّر", "يُضاعف"], 1, "داخل الخط."),
        Q("`prefetch(AUTOTUNE)`…", ["يخلط", "يجهّز الدفعة التالية أثناء تدريب الحالية", "يحفظ على القرص"], 1, "تداخل CPU/GPU."),
    ])
    takeaway("Dataset كسول: from_tensor_slices → map → shuffle(buffer ≥ n) → batch(B) → prefetch(AUTOTUNE). shuffle قبل batch دائمًا. fit يقبله مباشرة.")
    lesson_footer(LESSON, ["الخط القياسي وترتيبه.", "buffer_size وdrop_remainder وAUTOTUNE.", "الخطآن الشائعان."])
