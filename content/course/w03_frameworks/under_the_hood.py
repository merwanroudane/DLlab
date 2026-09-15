import streamlit as st

from components.callouts import common_mistake, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline

LESSON = Lesson(
    id="course.w03.under_the_hood",
    title_ar="تحت غطاء fit: GradientTape وtf.data والأجهزة — واللقطات",
    title_en="Under fit(): GradientTape, tf.data, Devices — and the Screens",
    module="course.w03",
    order=3,
    prerequisites=["course.w03.first_network", "foundations.frameworks.tensorflow.gradient_tape", "foundations.frameworks.tensorflow.tf_data"],
    objectives_ar=["إعادة كتابة خطوة واحدة من fit للشبكة الأولى بـ GradientTape وapply_gradients ومطابقتها بأرقام Keras.", "تغذية النموذج نفسه بـ tf.data (Dataset/batch/shuffle) بدل المصفوفات.", "قراءة معلومات الجهاز، ومعرفة أين تجد لقطات المخرجات الحقيقية (المعرض)."],
    terms=["gradient", "batch", "epoch"],
    difficulty="intermediate",
    summary_ar="fit = for epoch: for batch in tf.data: with GradientTape: forward, loss; grads; apply_gradients; metrics. الجهاز تلقائي. اللقطات في الوحدة 22.",
)

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf, keras
from keras import layers
from labs.tinynet import make_moons
keras.utils.set_random_seed(0)
X, y = make_moons(400, noise=0.25, seed=0); y = y[:, None]
ds = tf.data.Dataset.from_tensor_slices((X[:300], y[:300])).shuffle(300, seed=0).batch(32).prefetch(tf.data.AUTOTUNE)   # 1) tf.data
X_val, y_val = X[300:], y[300:]

model = keras.Sequential([layers.Input(shape=(2,)), layers.Dense(16, activation="relu"), layers.Dense(1, activation="sigmoid")])
loss_fn = keras.losses.BinaryCrossentropy(); opt = keras.optimizers.Adam(1e-2); acc = keras.metrics.BinaryAccuracy()

# 2) خطوة fit الواحدة مكتوبة يدويًا
xb, yb = next(iter(ds))
print("batch shapes:", tuple(xb.shape), tuple(yb.shape), "| device:", xb.device.split("/")[-1])
with tf.GradientTape() as tape:                                   # تسجيل
    out = model(xb, training=True)                                # أمامي
    loss = loss_fn(yb, out)                                       # خسارة
grads = tape.gradient(loss, model.trainable_variables)            # خلفي (الوحدة 14 آليًا)
print("loss before step:", round(float(loss), 4), "| grad shapes:", [tuple(g.shape) for g in grads])
w_before = float(model.layers[0].kernel[0, 0])
opt.apply_gradients(zip(grads, model.trainable_variables))        # تحديث
print(f"kernel[0,0]: {w_before:+.4f} -> {float(model.layers[0].kernel[0, 0]):+.4f} | loss after: {float(loss_fn(yb, model(xb))):.4f}")

# 3) حلقة كاملة صغيرة = fit(epochs=3)
for epoch in range(1, 4):
    acc.reset_state(); tot = 0.0; n = 0
    for xb, yb in ds:
        with tf.GradientTape() as tape:
            out = model(xb, training=True); loss = loss_fn(yb, out)
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        acc.update_state(yb, out); tot += float(loss) * len(xb); n += len(xb)
    vl = float(loss_fn(y_val, model(X_val, training=False)))
    print(f"epoch {epoch}/3 - loss: {tot / n:.4f} - accuracy: {float(acc.result()):.3f} - val_loss: {vl:.4f}")

# 4) نفس النموذج يقبل Dataset في fit مباشرة (بلا batch_size)
model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=["accuracy"])
h = model.fit(ds, epochs=2, validation_data=(X_val, y_val), shuffle=False, verbose=0)   # الخلط داخل الخط
print("fit(ds) ok -> keys:", list(h.history), "| GPUs:", tf.config.list_physical_devices("GPU"))'''


def render() -> None:
    lesson_header(LESSON)
    why("`fit` في الدرس السابق أخفى الوحدات 14 و15 و16 و17 في سطر. هنا نفتحه على **نفس نوع النموذج** ونطابق سطر السجل الذي يطبعه Keras بسطر نكتبه نحن — ثم نعيد الغطاء ونستخدم fit مع tf.data.")
    pipeline(["tf.data → batches", "GradientTape → grads", "optimizer.apply_gradients", "metrics", "validation", "= one fit() epoch"], active=1)
    code_lab(CodeLab(
        key="w03_hood", title_ar="خطوة fit يدويًا، ثم حلقة كاملة، ثم fit على Dataset", code=CODE, level="B",
        before=Before(goal_ar="رؤية ما يحدث في تحديث واحد (أشكال التدرجات، وزن قبل/بعد، خسارة قبل/بعد)، ثم ثلاث حقب بسطر سجل بشكل Keras، ثم fit على tf.data.", stage_ar="الأسبوع 03: تحت الغطاء.",
                      inputs_ar="moons (300 تدريب / 100 تحقق) كـ tf.data.Dataset.", expected_ar="دفعة (32, 2)/(32, 1) على CPU؛ 4 تدرجات بأشكال المعلمات؛ الوزن يتغير والخسارة تنخفض بعد الخطوة؛ ثلاثة أسطر سجل؛ fit(ds) يعمل.", prerequisites_ar="الوحدة 21: GradientTape، tf.data، الحلقة المخصصة."),
        explain=[("7", "خط بيانات كامل: from_tensor_slices → shuffle → batch → prefetch. هذا ما يفعله fit بالمصفوفات ضمنيًا."), ("14-22", "خطوة واحدة: دفعة → شريط → أمامي بـ training=True → خسارة → تدرج لكل معلمة → apply_gradients. الوزن يتغير والخسارة تنخفض على **نفس الدفعة**."),
                 ("25-33", "الحلقة كاملة: تصفير المقاييس، حلقة الدفعات، تجميع موزون، تحقق بـ training=False. السطر المطبوع = سطر fit."), ("36-38", "الغطاء مرة أخرى: compile + fit على Dataset (batch_size داخل الخط). الجهاز: [] هنا = CPU؛ في Colab مع GPU تُوضع الموترات عليه تلقائيًا بلا تغيير في الكود.")],
        run=run_printed(CODE),
        after_ar="- 4 تدرجات لأربع معلمات (kernel, bias × طبقتين) بنفس أشكالها: هذا هو `.grad` في PyTorch لكن كقيم مرجعة.\n- الخسارة بعد الخطوة أقل على نفس الدفعة — ليس ضمانًا للتعميم، لكنه دليل أن الاتجاه صحيح.\n- الأسطر الثلاثة تطابق شكل سجل Keras لأن المصدر واحد.",
    ))
    h2("أين اللقطات؟", "Where are the screens?")
    compare_table(["ما تريد رؤيته", "الصفحة"],
                  [("مخرج tf.Tensor، summary، سجل fit، evaluate/predict، رسم History، TensorBoard", "الوحدة 22: معرض TensorFlow وKeras"), ("موتر PyTorch، print(model)، سجل حلقة، CPU/GPU", "الوحدة 22: معرض PyTorch"), ("خطأ شكل وخطأ dtype/جهاز كما تظهر فعليًا", "الوحدة 22: معرض الأخطاء"), ("قبل/أثناء/بعد التدريب", "الوحدة 22: قبل/بعد"), ("لوحات TensorBoard مشروحة", "الوحدة 21: TensorBoard")],
                  ["rtl", "rtl"])
    with st.container(horizontal=True):
        st.button("المعرض — Keras/TensorFlow", icon=":material/photo_library:", on_click=go, args=("foundations.gallery.keras_tf",), key="w03_go_gal_k")
        st.button("المعرض — الأخطاء", icon=":material/bug_report:", on_click=go, args=("foundations.gallery.errors",), key="w03_go_gal_e")
        st.button("TensorBoard", icon=":material/monitoring:", on_click=go, args=("foundations.frameworks.tensorflow.tensorboard_errors",), key="w03_go_tb")
    intuition("Keras: fit تخفي الحلقة. TensorFlow: الحلقة بأدواته الأربع. PyTorch (الدرس التالي): الحلقة نفسها هي الأسلوب الافتراضي. ثلاث طبقات لنفس الرياضيات.")
    common_mistake("كتابة حلقة مخصصة في TensorFlow لمسألة قياسية «لأنها أوضح»: تخسر callbacks وHistory وأشرطة التقدم وتربح أخطاء. الحلقة المخصصة للحاجة غير القياسية؛ للتعلم اقرأها مرة كما هنا.")
    quiz("w03.hood", [
        Q("`tape.gradient(loss, model.trainable_variables)` يعيد…", ["عددًا", "قائمة تدرجات بأشكال المعلمات", "الخسارة"], 1, ""),
        Q("`model(xb, training=True)` مقابل `training=False`:", ["لا فرق", "يفعّل Dropout/إحصاءات BN الدفعية", "يحدّث الأوزان"], 1, ""),
        Q("`fit(ds)` مع Dataset مقسّم بـ batch(32):", ["مرّر batch_size=32 أيضًا", "لا تمرّر batch_size", "يفشل"], 1, ""),
        Q("مع GPU في Colab، كود Keras…", ["يُعاد كتابته", "لا يتغير", "يحتاج .to(device)"], 1, ""),
    ])
    takeaway("fit = tf.data + GradientTape + apply_gradients + metrics في حلقتين. رأيتها مرة؛ استخدم fit في العمل. اللقطات الحقيقية في الوحدة 22.")
    lesson_footer(LESSON, ["خطوة واحدة مكشوفة.", "حلقة كاملة تطابق سجل Keras.", "روابط اللقطات."])
