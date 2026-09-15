import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, intuition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="foundations.frameworks.tensorflow.tensors",
    title_ar="tf.Tensor: الإنشاء، الشكل، الرتبة، dtype، العمليات، البث، Variable، الأجهزة",
    title_en="tf.Tensor: Creation, Shape, Rank, dtype, Ops, Broadcasting, Variables & Devices",
    module="foundations.frameworks",
    parent="foundations.frameworks.tensorflow",
    order=14,
    prerequisites=["foundations.frameworks.tensorflow", "foundations.python.vectorization_broadcasting", "foundations.data.shape_axis_rank"],
    objectives_ar=["إنشاء موترات بطرق مختلفة وقراءة shape/rank/dtype/device.", "العمليات والبث كما في NumPy، مع قواعد dtype الصارمة في TensorFlow.", "الفرق بين tf.constant (ثابت) وtf.Variable (معلمة قابلة للتحديث)، ووضع الموترات على الأجهزة."],
    terms=["tensor", "shape", "rank", "dtype", "broadcasting", "axis"],
    labs=["labs.tf_tensor_explorer"],
    difficulty="intermediate",
    summary_ar="tf.constant/zeros/ones/random لإنشاء موتر ثابت؛ tf.Variable لمعلمة. shape وrank وdtype كما في NumPy، لكن TensorFlow لا يخلط dtypes تلقائيًا (cast صريح). البث بنفس قواعد NumPy. الجهاز خاصية للموتر.",
)

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf

# الإنشاء
a = tf.constant([[1, 2, 3], [4, 5, 6]])                 # int32 من أعداد صحيحة بايثون
b = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])     # float32 من كسور
z = tf.zeros((2, 3)); o = tf.ones((3,)); r = tf.random.normal((2, 3), seed=0)
n = tf.convert_to_tensor(np.arange(6).reshape(2, 3))    # من NumPy: يرث int64 من NumPy
for name, t in [("a", a), ("b", b), ("zeros", z), ("ones", o), ("normal", r), ("from numpy", n)]:
    print(f"{name:<11} shape={tuple(t.shape)!s:<8} rank={int(tf.rank(t))} dtype={t.dtype.name:<8} size={int(tf.size(t))}")

# العمليات والبث (نفس قواعد NumPy، وحدة 2)
print("b + ones(3,)        ->", (b + o).numpy().tolist(), "  (3,) بُثّت على (2,3)")
print("b * 2               ->", (b * 2).numpy().tolist())
print("tf.reduce_mean(b, 0)->", tf.reduce_mean(b, axis=0).numpy().tolist(), " (متوسط كل عمود)")
print("tf.matmul(b, tf.transpose(b)) ->", tf.matmul(b, tf.transpose(b)).numpy().tolist(), "(2,3)@(3,2)=(2,2)")
print("b[:, 1]             ->", b[:, 1].numpy().tolist(), "| tf.reshape(b, (3, 2)) ->", tf.reshape(b, (3, 2)).numpy().tolist())

# dtype صارم: لا خلط تلقائي
try:
    a + b
except Exception as e:
    print("a(int32) + b(float32) ->", type(e).__name__ + ":", str(e).split("[Op")[0].strip()[:90])
print("fix: tf.cast(a, tf.float32) + b ->", (tf.cast(a, tf.float32) + b).numpy().tolist())

# constant مقابل Variable
w = tf.Variable([[0.1, 0.2, 0.3]], name="w")
print("Variable:", w.name, w.shape, w.dtype.name, "trainable =", w.trainable)
w.assign_add([[1.0, 1.0, 1.0]])                          # تحديث في مكانه
print("after assign_add ->", w.numpy().round(2).tolist())
try:
    b.assign_add(1.0)
except AttributeError as e:
    print("constant.assign_add ->", "AttributeError: EagerTensor has no attribute 'assign_add' (constants are immutable)")

# الأجهزة
print("device of b:", b.device.split("/")[-1], "| GPUs:", tf.config.list_physical_devices("GPU"))
with tf.device("/CPU:0"):                                 # وضع صريح (مع GPU: "/GPU:0")
    c = tf.ones((2, 2))
print("explicit placement ->", c.device.split("/")[-1])'''


def render() -> None:
    lesson_header(LESSON)
    definition("**`tf.Tensor`**: مصفوفة متعددة الأبعاد **غير قابلة للتغيير** بشكل وdtype وجهاز. **`tf.Variable`**: موتر **قابل للتغيير في مكانه** (`assign`, `assign_add`, `assign_sub`) يمثّل المعلمات ويُتتبَّع تلقائيًا بواسطة `GradientTape`. الرتبة = عدد المحاور، الشكل = طول كل محور، dtype = نوع كل عنصر (وحدات 1 و4).")
    code_lab(CodeLab(
        key="tf_tensors", title_ar="إنشاء، فحص، عمليات، dtype صارم، Variable، الجهاز", code=CODE, level="B",
        before=Before(goal_ar="لمس كل خاصية للموتر في TensorFlow ومقارنتها بما تعرفه من NumPy، مع كشف الفرقين المهمين: dtype الصارم وعدم قابلية الثابت للتغيير.", stage_ar="TensorFlow: الموترات.",
                      inputs_ar="موترات صغيرة يدوية.", expected_ar="جدول الخصائص لست طرق إنشاء؛ بث ناجح؛ خطأ عند جمع int32 مع float32 وإصلاحه بـ cast؛ Variable يتحدث بـ assign_add والثابت لا؛ الجهاز CPU:0.",
                      objects_ar="`tf.constant`, `tf.zeros`, `tf.ones`, `tf.random.normal`, `tf.Variable`, `tf.device`."),
        explain=[("4-10", "ست طرق إنشاء. لاحظ dtype المستنتج: أعداد بايثون الصحيحة → int32، الكسور → float32، من NumPy → int64 (يرث NumPy). هذا مصدر أخطاء لاحقًا."),
                 ("13-17", "البث (`(3,)` على `(2,3)`)، الاختزال على محور، الضرب المصفوفي والفهرسة وreshape — كل قواعد الوحدتين 2 و4 تنطبق."),
                 ("20-24", "**الفرق الأول عن NumPy**: TensorFlow لا يرقّي int32 إلى float32 تلقائيًا؛ يرفع خطأ. الحل `tf.cast`."),
                 ("27-34", "**الفرق الثاني**: الثابت لا يُعدَّل؛ Variable يُعدَّل في مكانه بـ assign_*. أوزان Keras كلها Variables."), ("37-40", "الجهاز خاصية للموتر؛ `tf.device` يضع الموترات صراحةً. بلا GPU كل شيء على CPU:0.")],
        run=run_printed(CODE),
        after_ar="- `from numpy … int64`: عند تمرير مصفوفات NumPy للنموذج حوّلها إلى float32 أولًا (`astype('float32')`) لتجنب أخطاء dtype وضعف الذاكرة.\n- خطأ `a + b` هو نفسه الذي ستراه عند خلط y (int) مع مخرج النموذج (float) في خسارة يدوية.\n- `trainable = True` هو ما يجعل GradientTape يراقب المتغير تلقائيًا.",
    ))
    st.button("افتح مستكشف موترات TensorFlow", icon=":material/science:", type="primary", on_click=go, args=("labs.tf_tensor_explorer",), key="tf_tensors_lab")
    h2("NumPy مقابل TensorFlow: ما الذي يتغير؟", "NumPy vs TensorFlow")
    compare_table(["الجانب", "NumPy", "TensorFlow"],
                  [("الإنشاء", "`np.array`, `np.zeros`, `rng.normal`", "`tf.constant`, `tf.zeros`, `tf.random.normal`"), ("الشكل/الرتبة", "`.shape`, `.ndim`", "`.shape`, `tf.rank(x)` / `.ndim`"), ("dtype الافتراضي", "float64 / int64", "float32 / int32"),
                   ("خلط dtypes", "ترقية تلقائية", "**خطأ**: cast صريح"), ("التغيير في مكانه", "أي مصفوفة", "`tf.Variable` فقط"), ("الجهاز", "CPU دائمًا", "خاصية للموتر؛ GPU تلقائي إن وُجد"),
                   ("الاشتقاق", "لا", "`GradientTape` على Variables"), ("التحويل", "—", "`x.numpy()` ↔ `tf.convert_to_tensor(a)`")],
                  ["rtl", "code", "code"])
    intuition("فكّر في `tf.Tensor` كـ ndarray بثلاث إضافات: جهاز، تسجيل للاشتقاق، وصرامة في dtype. وفي `tf.Variable` كـ ndarray «مسجّل رسميًا» كمعلمة. كل ما تعرفه عن الأشكال والمحاور والبث ينتقل كما هو.")
    debugging_note("`InvalidArgumentError: cannot compute AddV2 as input #1 was expected to be a int32 tensor but is a float tensor` = خلط dtypes. ابحث عن مصدر int (عادةً y من `np.array([0,1,2])` أو من pandas) وحوّله: `tf.cast(y, tf.float32)` أو `y.astype('float32')` قبل الدخول.")
    common_mistake("`x = tf.constant(...)` ثم `x[0, 0] = 5`: `TypeError: 'EagerTensor' object does not support item assignment`. الموترات الثابتة لا تُعدَّل جزئيًا؛ استخدم `tf.Variable` مع `x[0, 0].assign(5)`، أو أنشئ موترًا جديدًا (`tf.where`, `tf.tensor_scatter_nd_update`).")
    quiz("tf.tensors", [
        Q("`tf.constant([1, 2])` dtype:", ["float32", "int32", "int64"], 1, "أعداد بايثون الصحيحة → int32."),
        Q("`int32 + float32` في TensorFlow…", ["يرقّي تلقائيًا", "يرفع خطأ", "يعطي int32"], 1, "cast صريح."),
        Q("لتحديث قيمة في مكانها تحتاج…", ["tf.constant", "tf.Variable", "tf.zeros"], 1, "assign."),
        Q("`tf.reduce_mean(b, axis=0)` على (2,3) يعطي شكل", ["(2,)", "(3,)", "()"], 1, "يختزل المحور 0."),
    ])
    takeaway("tf.Tensor = ndarray + جهاز + تسجيل + dtype صارم؛ ثابت. tf.Variable = معلمة قابلة للتحديث. float32/int32 افتراضيًا. cast صريح دائمًا. البث والمحاور كما في NumPy.")
    lesson_footer(LESSON, ["ست طرق إنشاء وخصائص أربع.", "الفرقان عن NumPy.", "Variable للمعلمات والأجهزة."])
