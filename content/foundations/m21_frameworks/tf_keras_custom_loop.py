import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.frameworks.tensorflow.keras_custom_loop",
    title_ar="علاقة TensorFlow بـ Keras، حلقة تدريب مخصصة (تعميق)، والحفظ/التصدير",
    title_en="TensorFlow ↔ Keras, a Custom Training Loop (Deep Dive) & Saving/Export",
    module="foundations.frameworks",
    parent="foundations.frameworks.tensorflow",
    order=17,
    prerequisites=["foundations.frameworks.tensorflow.gradient_tape", "foundations.frameworks.tensorflow.tf_data", "foundations.frameworks.keras.fit"],
    objectives_ar=["رؤية fit مكتوبًا يدويًا بـ GradientTape وtf.data وapply_gradients — ومطابقته لخطوات محرك fit.", "معرفة متى تحتاج الحلقة المخصصة (خسائر مركّبة، تدرجات معدَّلة، شبكات متعددة) ومتى لا.", "خريطة الحفظ: .keras، weights، SavedModel export."],
    terms=["epoch", "batch", "gradient", "optimizer"],
    difficulty="advanced",
    summary_ar="حلقة مخصصة = for epoch: for batch in ds: with tape: forward, loss; grads = tape.gradient; optimizer.apply_gradients; metrics.update_state. هذا ما يفعله fit. الحفظ: model.save('.keras') للكل، export للنشر.",
)

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf, keras
from keras import layers
keras.utils.set_random_seed(0)
rng = np.random.default_rng(0)
X = rng.normal(size=(400, 3)).astype("float32"); y = (X[:, 0] * X[:, 1] > 0).astype("float32")[:, None]
ds = tf.data.Dataset.from_tensor_slices((X[:320], y[:320])).shuffle(320, seed=0).batch(32)
X_val, y_val = X[320:], y[320:]

model = keras.Sequential([layers.Input(shape=(3,)), layers.Dense(16, activation="relu"), layers.Dense(1, activation="sigmoid")])
loss_fn = keras.losses.BinaryCrossentropy()
optimizer = keras.optimizers.Adam(1e-2)
acc = keras.metrics.BinaryAccuracy(); loss_avg = keras.metrics.Mean()

for epoch in range(1, 4):                                     # حلقة الحقب
    acc.reset_state(); loss_avg.reset_state()
    for step, (xb, yb) in enumerate(ds, start=1):             # حلقة الدفعات
        with tf.GradientTape() as tape:                       # تسجيل
            y_hat = model(xb, training=True)                  # أمامي (training=True لـ Dropout/BN)
            loss = loss_fn(yb, y_hat)                         # خسارة الدفعة
        grads = tape.gradient(loss, model.trainable_variables)          # خلفي
        optimizer.apply_gradients(zip(grads, model.trainable_variables))  # تحديث
        loss_avg.update_state(loss); acc.update_state(yb, y_hat)          # تجميع المقاييس
    val_pred = model(X_val, training=False)                   # تحقق: بلا شريط، بلا تحديث
    val_loss = float(loss_fn(y_val, val_pred)); val_acc = float(tf.reduce_mean(tf.cast((val_pred > 0.5) == (y_val > 0.5), tf.float32)))
    print(f"epoch {epoch}: {step} steps - loss {float(loss_avg.result()):.4f} - accuracy {float(acc.result()):.3f} - val_loss {val_loss:.4f} - val_accuracy {val_acc:.3f}")
print("grad shapes:", [tuple(g.shape) for g in grads], "| n trainable vars:", len(model.trainable_variables))

# نفس النموذج يمكنه أيضًا استخدام compile/fit — الحلقة المخصصة ليست بديلًا عن Keras بل نزول تحتها
model.compile(optimizer=keras.optimizers.Adam(1e-2), loss=loss_fn, metrics=["accuracy"])
print("evaluate via Keras:", [round(v, 4) for v in model.evaluate(X_val, y_val, verbose=0)])'''

SAVE = '''model.save("model.keras")                         # الكل: بنية + أوزان + compile  → keras.saving.load_model
model.save_weights("model.weights.h5")            # الأوزان فقط                     → model.load_weights (نفس البنية)
model.export("saved_model_dir")                   # SavedModel للاستدلال خارج بايثون (TF Serving, TFLite, TF.js)
tf.saved_model.load("saved_model_dir")            # تحميل الرسم المصدَّر (بلا fit؛ استدلال فقط)'''


def render() -> None:
    lesson_header(LESSON)
    h2("العلاقة", "The relationship")
    definition("**Keras فوق TensorFlow**: `keras.layers.Dense` تخزّن أوزانها كـ `tf.Variable`؛ `model(x)` ينفذ عمليات `tf.*`؛ `fit` يفتح `tf.GradientTape`، يستدعي `tape.gradient(loss, model.trainable_variables)` ثم `optimizer.apply_gradients`، ويمر على البيانات كـ `tf.data.Dataset`. لا يوجد في fit شيء لا يمكنك كتابته بنفسك بهذه الأدوات الأربع.")
    compare_table(["خطوة محرك fit (درس fit)", "ما تكتبه في الحلقة المخصصة"],
                  [("Dataset / Arrays → Batches", "`tf.data.Dataset.from_tensor_slices(...).shuffle().batch()`"), ("Epoch loop / Batch loop", "`for epoch in …: for xb, yb in ds:`"), ("Forward", "`y_hat = model(xb, training=True)`"), ("Loss", "`loss = loss_fn(yb, y_hat)`"),
                   ("Autodiff", "`with tf.GradientTape() as tape:` … `tape.gradient(loss, model.trainable_variables)`"), ("Optimizer update", "`optimizer.apply_gradients(zip(grads, vars))`"), ("Metric update", "`metric.update_state(yb, y_hat)` … `metric.result()`"), ("Validation", "`model(X_val, training=False)` بلا شريط"), ("History", "قائمة/قاموس تملؤه بنفسك")],
                  ["rtl", "code"])
    why("متى تنزل إلى الحلقة المخصصة؟ عندما تحتاج ما لا تعبّر عنه معاملات fit: خسارة تجمع عدة مخرجات بأوزان متغيرة، تعديل التدرجات قبل التطبيق (قصّ مخصص، تدرجات لشبكتين متنافستين كما في GAN)، خطوات تدريب متعددة لكل دفعة، أو تسجيل تشخيصي لكل معلمة. في المسار الأساسي: **لا تحتاجها** — لكن قراءتها مرة تجعل fit شفافًا إلى الأبد.")
    code_lab(CodeLab(
        key="tf_custom_loop", title_ar="fit مكتوبًا يدويًا: نفس الخطوات، نفس المخرجات", code=CODE, level="B",
        before=Before(goal_ar="كتابة حلقة تدريب صريحة بأدوات TensorFlow/Keras الأربع وإنتاج سطر سجل مطابق لشكل سطر fit، ثم إثبات أن نفس النموذج يعمل مع compile/evaluate.", stage_ar="TensorFlow: حلقة مخصصة (تعميق).",
                      inputs_ar="400 ملاحظة × 3، هدف ثنائي؛ 320 تدريب / 80 تحقق.", expected_ar="ثلاثة أسطر بشكل سطر fit (10 خطوات لكل حقبة)، أشكال التدرجات = أشكال المتغيرات (4)، وevaluate يعمل.",
                      objects_ar="`tf.data.Dataset`, `tf.GradientTape`, `keras.losses.BinaryCrossentropy`, `keras.optimizers.Adam`, `keras.metrics.*`."),
        explain=[("7-8", "خط البيانات (الدرس السابق) ومجموعة تحقق كمصفوفات."), ("10-13", "النموذج والخسارة والمحسّن والمقاييس **ككائنات منفصلة** — ما كان compile يربطه."),
                 ("15-16", "حلقة الحقب وتصفير مجمّعات المقاييس (كما يفعل fit في بداية كل حقبة)."), ("17-23", "حلقة الدفعات: الخطوات الخمس — تسجيل، أمامي بـ `training=True`، خسارة، تدرج بالنسبة لـ `trainable_variables`، تطبيق. ثم تحديث المقاييس."),
                 ("24-26", "التحقق: أمامي فقط بـ `training=False` وبلا شريط. السطر المطبوع يحاكي سطر fit."), ("27", "قائمة التدرجات تطابق قائمة المتغيرات واحدًا لواحد."), ("30-31", "الحلقة المخصصة لا تمنع استخدام Keras لاحقًا على نفس النموذج.")],
        run=run_printed(CODE),
        after_ar="- قارن السطر المطبوع بسطر fit في «مستكشف المخرجات»: نفس الحقول لأن المصدر نفسه.\n- `training=True/False` هنا مسؤوليتك — في fit/evaluate تلقائي. هذا هو أول ما يُنسى في الحلقات المخصصة.\n- الفرق بين هذه الحلقة وحلقة PyTorch (الصفحات التالية): هناك هذا هو الأسلوب **الافتراضي**.",
    ))
    research_note("**تعميق أعمق**: بدل الحلقة الكاملة يمكنك تجاوز `train_step` فقط في صنف يرث `keras.Model` — فتحتفظ بـ fit وcallbacks وHistory وتغيّر الخطوة الواحدة. وتزيين الخطوة بـ `@tf.function` يجمّعها كرسم للسرعة.")
    h2("الحفظ والتصدير", "Saving & export")
    st.code(SAVE, language="python")
    compare_table(["الصيغة", "يحوي", "تحميل", "الاستخدام"],
                  [("`.keras`", "بنية + أوزان + compile + حالة المحسّن", "`keras.saving.load_model`", "الافتراضي داخل بايثون/Keras؛ استئناف التدريب"), ("`.weights.h5`", "أوزان فقط", "`load_weights` على نفس البنية", "نقاط حفظ، نقل أوزان"),
                   ("SavedModel (`export`)", "رسم استدلال TensorFlow", "`tf.saved_model.load`، TF Serving، TFLite", "النشر؛ لا يمكن fit عليه")],
                  ["code", "rtl", "code", "rtl"])
    intuition("`.keras` للباحث (يعيد النموذج كما تركته)، SavedModel للمهندس (رسم ثابت يعمل بلا بايثون). في المقرر ستحتاج الأولى فقط.")
    common_mistake("نسيان `training=True` في الحلقة المخصصة: Dropout لا يعمل وBN تستخدم المتحركة أثناء التدريب. أو العكس: `training=True` في التحقق فتتغير النتائج بين استدعاء وآخر.")
    quiz("tf.custom", [
        Q("الأدوات الأربع التي يبنى منها fit:", ["Dataset, Tape, apply_gradients, metrics", "Colab, GPU, NumPy, pandas", "summary, predict, evaluate, save"], 0, "الحلقة المخصصة."),
        Q("`model(xb, training=True)` مقابل `training=False`:", ["لا فرق", "يفعّل Dropout/إحصاءات BN الدفعية", "يحدّث الأوزان"], 1, "وضع الطبقات."),
        Q("متى تحتاج الحلقة المخصصة؟", ["دائمًا", "عند خسائر/تدرجات/خطوات غير قياسية", "أبدًا"], 1, "تحكم غير قياسي."),
        Q("`model.export()` ينتج…", [".keras قابل لـ fit", "SavedModel للاستدلال", "أوزانًا فقط"], 1, "نشر."),
    ])
    takeaway("fit = Dataset + Tape + apply_gradients + metrics في حلقتين. اكتبها مرة لتفهم، واستخدم fit في العمل ما لم تحتج تحكمًا غير قياسي. احفظ .keras للبحث، وexport للنشر.")
    lesson_footer(LESSON, ["جدول المطابقة بين محرك fit والحلقة.", "الحلقة المخصصة كاملة.", "ثلاث صيغ حفظ."])
