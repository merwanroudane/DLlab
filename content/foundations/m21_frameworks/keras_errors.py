import streamlit as st

from components.callouts import debugging_note, intuition, practical_note, takeaway, why
from components.code_lab import exec_source
from components.comparison import compare_table
from components.diagnostics import Problem, problem_card
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.frameworks.keras.errors",
    title_ar="أخطاء Keras الشائعة: الأشكال، الخسارة/التنشيط، ترميز الهدف، وسلوك GPU",
    title_en="Common Keras Errors: Shapes, Loss/Activation Mismatch, Target Encoding & GPU Behaviour",
    module="foundations.frameworks",
    parent="foundations.frameworks.keras",
    order=11,
    prerequisites=["foundations.frameworks.keras.output_explorer", "foundations.loss.loss_activation_compatibility", "foundations.prep.encoding"],
    objectives_ar=["قراءة رسائل الخطأ الحقيقية من Keras/TensorFlow وتحديد السطر المسؤول.", "تصحيح أخطاء الأشكال، عدم توافق الخسارة والتنشيط، وترميز الهدف الخاطئ — وكشف الأخطاء الصامتة.", "معرفة ما يراه المتعلم عن GPU: كيف يعرف إن كان يستخدمه، ولماذا لا يغيّر الكود."],
    terms=["shape", "batch_dimension", "cross_entropy", "dtype"],
    difficulty="intermediate",
    summary_ar="أخطاء الأشكال تقرأ من آخر سطر: expected … but received …. الخسارة الخاطئة قد تعطي خطأ صريحًا (rank) أو تعمل بصمت بنتائج خاطئة. GPU لا يغيّر الكود؛ تحقق منه بـ tf.config.list_physical_devices.",
)

ERR_SHAPE = """ValueError: Exception encountered when calling Sequential.call().

Input 0 with name 'None' of layer 'dense' is incompatible with the layer:
expected axis -1 of input shape to have value 5, but received input with shape (32, 4)

Arguments received by Sequential.call():
  • inputs=tf.Tensor(shape=(32, 4), dtype=float32)
  • training=True"""

ERR_RANK = """ValueError: Arguments `target` and `output` must have the same rank (ndim).
Received: target.shape=(32,), output.shape=(32, 3)"""

ERR_VECTOR = """ValueError: Invalid input shape for input Tensor("data:0", shape=(4,), dtype=float32).
Expected shape (None, 4), but input has incompatible shape (4,)"""

ERR_DTYPE = """ValueError: Invalid dtype: object"""

SILENT = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, keras
from keras import layers
keras.utils.set_random_seed(0)
rng = np.random.default_rng(0)
X = rng.normal(size=(300, 4)).astype("float32")
y = rng.integers(0, 3, 300)                                   # ثلاث فئات: 0, 1, 2
# خطأ صامت: مخرج sigmoid بوحدة واحدة + binary_crossentropy مع هدف من 3 فئات
bad = keras.Sequential([layers.Input(shape=(4,)), layers.Dense(1, activation="sigmoid")])
bad.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
h = bad.fit(X, y.astype("float32"), epochs=3, verbose=0)      # لا خطأ!
print("silent wrong model  -> loss:", np.round(h.history["loss"], 3), "accuracy:", np.round(h.history["accuracy"], 3))
print("   predictions range:", bad.predict(X[:5], verbose=0).ravel().round(2), "(cannot represent class 2)")
# الصحيح
good = keras.Sequential([layers.Input(shape=(4,)), layers.Dense(3, activation="softmax")])
good.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h2 = good.fit(X, y, epochs=3, verbose=0)
print("correct model       -> loss:", np.round(h2.history["loss"], 3), "accuracy:", np.round(h2.history["accuracy"], 3))
print("   predict shape:", good.predict(X[:5], verbose=0).shape, "argmax:", good.predict(X[:5], verbose=0).argmax(1))'''

GPU_CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
print("TensorFlow", tf.__version__)
print("GPUs visible :", tf.config.list_physical_devices("GPU"))
print("CPUs visible :", tf.config.list_physical_devices("CPU"))
x = tf.constant([[1.0, 2.0]])
print("tensor device:", x.device)          # /job:localhost/replica:0/task:0/device:CPU:0 على هذه الآلة'''


def _silent_experiment() -> None:
    st.markdown("نفس البيانات بثلاث فئات: نموذج خاطئ لا يرفع أي خطأ، ونموذج صحيح.")
    st.code(SILENT, language="python")
    st.code(exec_source(SILENT).rstrip(), language="text")


def render() -> None:
    lesson_header(LESSON)
    why("رسائل TensorFlow طويلة ومخيفة، لكن **السطر المفيد دائمًا هو الأخير** أو الذي يحوي `expected … but received …`. والأخطر من الخطأ الصريح هو **الخطأ الصامت**: كود يعمل ويعطي أرقامًا خاطئة. هذه الصفحة تعلّمك القراءة وتكشف الصامت.")
    h2("كيف تقرأ رسالة خطأ من Keras", "How to read a Keras error")
    st.markdown("""
1. **آخر سطر أولًا**: نوع الخطأ (`ValueError`, `InvalidArgumentError`) والرسالة. تجاهل الـ traceback الطويل في البداية.
2. ابحث عن **شكلين**: `expected (None, 5)` مقابل `received (32, 4)`. الأول ما يظنه النموذج، الثاني ما أعطيته.
3. ابحث عن **اسم الطبقة** (`layer 'dense'`) أو **العقدة** (`compile_loss/sparse_categorical_crossentropy`) — تحدد أين وقع الخطأ: المدخل، طبقة وسطى، أم الخسارة.
4. `Graph execution error` = الخطأ وقع داخل التنفيذ في TensorFlow؛ اقرأ السطر `Detected at node …` ثم الرسالة بعده.
5. اطبع `X.shape, X.dtype, y.shape, y.dtype, np.unique(y)` و`model.summary()` — 90% من الأخطاء تُحل بهذه السطر.
""")
    h2("1) عدم توافق شكل المدخل", "1) Input shape mismatch")
    problem_card(Problem(
        key="k_shape", name_ar="شكل المدخل لا يطابق Input", name_en="Input shape mismatch",
        description_ar="النموذج بُني بـ `Input(shape=(5,))` والبيانات بـ 4 خصائص (أو العكس).",
        symptoms_ar=["`fit`/`predict` يرفع ValueError فورًا قبل أي حقبة."], sees_ar=["السطر `expected axis -1 of input shape to have value 5, but received input with shape (32, 4)`."],
        possible_causes_ar=["خصائص أُضيفت/حُذفت بعد بناء النموذج.", "one-hot زاد عدد الأعمدة.", "نسيان Flatten لصورة."], root_causes_ar=["Input يصف ملاحظة واحدة؛ آخر بُعد يجب أن يساوي عدد الأعمدة الفعلي."],
        diagnosis_ar=["اطبع `X.shape`.", "قارن آخر بُعد بـ `Input(shape=…)` في summary.", "حدد أيهما الصحيح (البيانات عادةً)."], evidence_ar=["`X.shape[1] != Input shape[-1]`."],
        fixes_ar=["`Input(shape=(X.shape[1],))` — اربط الشكل بالبيانات لا برقم مكتوب يدويًا.", "للصور: `Flatten()` أو `Input(shape=(28, 28, 1))`."], error_text=ERR_SHAPE,
        wrong_code="model = keras.Sequential([layers.Input(shape=(5,)), layers.Dense(3)])\nmodel.fit(X, y)   # X.shape == (300, 4)", correct_code="model = keras.Sequential([layers.Input(shape=(X.shape[1],)), layers.Dense(3)])",
        misdiagnosis_ar=["«32 في الرسالة هو المشكلة» — لا، 32 هو batch_size؛ المشكلة في 4 مقابل 5."],
    ))
    h2("2) الخسارة وترميز الهدف", "2) Loss vs target encoding")
    problem_card(Problem(
        key="k_rank", name_ar="categorical_crossentropy مع هدف أعداد صحيحة", name_en="Loss/target rank mismatch",
        description_ar="الخسارة `categorical_crossentropy` تتوقع one-hot بشكل (batch, K)، والهدف أعداد صحيحة بشكل (batch,).",
        symptoms_ar=["ValueError عند أول دفعة."], sees_ar=["`target.shape=(32,), output.shape=(32, 3)` … `must have the same rank`."],
        possible_causes_ar=["اختيار الخسارة دون النظر إلى ترميز y."], root_causes_ar=["كل خسارة تصنيف لها ترميز هدف محدد (وحدة 13): sparse ↔ أعداد صحيحة، categorical ↔ one-hot، binary ↔ 0/1 بعمود واحد."],
        diagnosis_ar=["اطبع `y.shape` و`np.unique(y)`.", "اطبع شكل مخرج النموذج من summary.", "طابق الثلاثة: الخسارة، ترميز y، وحدات/تنشيط المخرج."], evidence_ar=["y أحادي البعد مع categorical_crossentropy."],
        fixes_ar=["الأسهل: `loss='sparse_categorical_crossentropy'` وإبقاء y أعدادًا صحيحة.", "أو `y = keras.utils.to_categorical(y, K)` مع `categorical_crossentropy`."], error_text=ERR_RANK,
        wrong_code="model.compile(loss='categorical_crossentropy')\nmodel.fit(X, y)          # y: [0, 2, 1, ...]", correct_code="model.compile(loss='sparse_categorical_crossentropy')\nmodel.fit(X, y)          # y: [0, 2, 1, ...]",
        related_ar=["الخطأ الصامت أدناه: نفس العائلة لكن بلا رسالة."],
    ))
    problem_card(Problem(
        key="k_silent", name_ar="خطأ صامت: sigmoid + binary_crossentropy لثلاث فئات", name_en="Silent loss/activation mismatch",
        description_ar="لا توجد رسالة خطأ. النموذج يتدرّب، الخسارة تنخفض قليلًا، والدقة عالقة — لأن وحدة sigmoid واحدة لا يمكنها تمثيل 3 فئات وbinary_crossentropy يعامل الهدف 2 كـ«احتمال 2».",
        symptoms_ar=["دقة عالقة قرب نسبة فئة واحدة.", "خسارة ثنائية بقيم غريبة (> 1 أحيانًا)."], sees_ar=["سجل fit يبدو «طبيعيًا» لكن الأداء سيئ ولا يتحسن."],
        possible_causes_ar=["نسخ قالب تصنيف ثنائي إلى مسألة متعددة الفئات."], root_causes_ar=["عدم فحص `np.unique(y)` قبل اختيار المخرج والخسارة."],
        diagnosis_ar=["`np.unique(y)` — كم فئة؟", "وحدات المخرج = 1 وعدد الفئات > 2؟ خطأ صامت.", "شغّل التجربة أدناه وقارن."], evidence_ar=["Dense(1, sigmoid) مع 3 قيم مميزة في y."],
        fixes_ar=["`Dense(K, activation='softmax')` + `sparse_categorical_crossentropy`."], experiment=_silent_experiment,
        checklist_ar=["فحصت `np.unique(y)` قبل بناء المخرج.", "وحدات المخرج = عدد الفئات (أو 1 للثنائي).", "الخسارة تطابق التنشيط وترميز y."],
        challenge=[Q("ثنائي 0/1 بعمود واحد: الخسارة والتنشيط", ["softmax + sparse", "sigmoid + binary_crossentropy", "بلا تنشيط + mse"], 1, "الثنائي.")],
    ))
    h2("3) أخطاء الشكل الأخرى وdtype", "3) Other shape errors & dtype")
    compare_table(["الرسالة (السطر المهم)", "السبب", "الحل"],
                  [(ERR_VECTOR.splitlines()[-1], "ملاحظة واحدة مُررت كمتجه (4,) بدل (1, 4)", "`x[None, :]` أو `x.reshape(1, -1)`"),
                   (ERR_DTYPE, "مصفوفة نصوص/كائنات (عمود فئوي غير مرمَّز) مُررت للنموذج", "رمّز الفئوي (one-hot/ordinal، وحدة 8) وحوّل إلى float32"),
                   ("Output Shape: ?   Param #: 0 (unbuilt)", "Sequential بلا Input فلم تُبنَ الأوزان", "أضف `layers.Input(shape=…)` أولًا"),
                   ("InvalidArgumentError: … Received a label value of 2 which is outside the valid range of [0, 1)", "sparse_categorical_crossentropy مع وحدة مخرج واحدة والهدف يحوي 2", "وحدات المخرج = عدد الفئات"),
                   ("loss: nan", "انفجار عددي: η كبير، مدخل غير محجّم، log(0)", "قلّل η، حجّم المدخل (وحدة 8)، `TerminateOnNaN`، القصّ (وحدة 14)")],
                  ["code", "rtl", "rtl"])
    debugging_note("قاعدة ذهبية: قبل `fit` اطبع سطرًا واحدًا: `print(X.shape, X.dtype, y.shape, y.dtype, np.unique(y)[:10])`. ثم `model.summary()`. طابق آخر بُعد لـ X مع Input، ووحدات المخرج مع عدد الفئات، والخسارة مع ترميز y.")
    h2("4) GPU من منظور المتعلم", "4) GPU from the learner's perspective")
    st.markdown("""
- **الكود لا يتغير**: Keras/TensorFlow يضعان الموترات والحساب على GPU تلقائيًا إن وُجد ومُثبَّت الدعم (CUDA). لا تحتاج `to(device)` كما في PyTorch.
- **كيف أعرف؟** `tf.config.list_physical_devices('GPU')` — قائمة فارغة = لا GPU (كما على هذه الآلة). في Colab: القائمة ← Runtime ← Change runtime type ← GPU.
- **ماذا يتغير؟** السرعة (10–100× للشبكات الكبيرة)، لا الدقة. النتائج قد تختلف قليلًا في الكسور العشرية (ترتيب الجمع غير الحتمي).
- **رسائل مربكة**: `Could not load dynamic library 'cudart64_…'` = TensorFlow لم يجد CUDA؛ سيعمل على CPU، وهذا **تحذير لا خطأ**.
- **OOM** (`ResourceExhaustedError: OOM when allocating tensor`): ذاكرة GPU امتلأت — قلّل `batch_size` أو حجم النموذج (الأسبوع 13).
""")
    st.code(GPU_CODE, language="python")
    st.code(exec_source(GPU_CODE).rstrip(), language="text")
    practical_note("على هذه الآلة لا يوجد GPU، فكل أمثلة الوحدة تعمل على CPU — وهي مصممة صغيرة لذلك. في Colab مع GPU تعمل نفس الملفات دون تعديل.")
    intuition("الأخطاء الصريحة هدية: تخبرك أين المشكلة. الأخطاء الصامتة هي ما يجب أن تخافه — والدفاع الوحيد هو **الفحص قبل fit** والمقارنة بخط أساس (وحدة 7).")
    quiz("keras.errors", [
        Q("`expected axis -1 … value 5, but received (32, 4)`: المشكلة في", ["batch_size 32", "عدد الخصائص 4 مقابل Input 5", "الخسارة"], 1, "آخر بُعد."),
        Q("`target.shape=(32,), output.shape=(32, 3)` … same rank:", ["استخدم sparse_categorical_crossentropy", "غيّر batch_size", "أضف Dropout"], 0, "ترميز الهدف."),
        Q("Dense(1, sigmoid) + binary_crossentropy مع y ∈ {0,1,2}…", ["يرفع خطأ", "يعمل بصمت بنتائج خاطئة", "يصحح نفسه"], 1, "صامت."),
        Q("`list_physical_devices('GPU')` تعيد []:", ["الكود سيفشل", "سيعمل على CPU (أبطأ فقط)", "يجب إعادة كتابة النموذج"], 1, "لا تغيير في الكود."),
    ])
    takeaway("اقرأ آخر سطر، طابق الشكلين، حدد الطبقة. قبل fit: اطبع الأشكال وdtype وunique(y) وsummary. الخسارة ↔ التنشيط ↔ ترميز الهدف ثلاثي لا ينفصل. GPU يغيّر السرعة لا الكود.")
    lesson_footer(LESSON, ["قراءة الرسالة في 5 خطوات.", "أربعة أخطاء صريحة وواحد صامت.", "GPU: كيف تعرف وماذا يتغير."])
