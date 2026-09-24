import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, definition, intuition, math_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import worked_steps
from components.quiz import Q, quiz
from content.course.w03_frameworks._viz import shuffle_sim, shuffle_svg, tape_numbers, tape_svg
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline

LESSON = Lesson(
    id="course.w03.under_the_hood",
    title_ar="تحت غطاء fit: GradientTape وtf.data والأجهزة — واللقطات",
    title_en="Under fit(): GradientTape, tf.data & Devices — and the Screens",
    module="course.w03",
    order=3,
    prerequisites=["course.w03.first_network", "foundations.frameworks.tensorflow.gradient_tape", "foundations.frameworks.tensorflow.tf_data"],
    objectives_ar=[
        "مشاهدة GradientTape وهو يسجّل الرسم الحسابي أماميًا ثم يمرر التدرجات عكسيًا بأرقام دقيقة.",
        "إعادة كتابة خطوة واحدة من fit بـ GradientTape وapply_gradients ومطابقتها بأرقام Keras.",
        "فهم خط tf.data (shuffle بمخزن، batch، prefetch) بمحاكاة عنصرًا عنصرًا، ثم تغذية النموذج به.",
        "قراءة معلومات الجهاز، ومعرفة أين تجد لقطات المخرجات الحقيقية (المعرض).",
    ],
    terms=["gradient", "batch", "epoch", "gradient_tape", "autodiff", "chain_rule", "tensorflow", "gpu", "optimizer"],
    labs=["labs.gradient_tape_lab", "labs.chain_rule_lab"],
    difficulty="intermediate",
    summary_ar="fit = for epoch: for batch in tf.data: with GradientTape: forward, loss; grads = tape.gradient; apply_gradients; metrics. الشريط يسجّل أماميًا ويشتق عكسيًا بقاعدة السلسلة. الجهاز تلقائي. اللقطات في الوحدة 22.",
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

    # ------------------------------------------------------------------ GradientTape graph
    h2("GradientTape: شريط يسجّل ثم يُعاد عكسيًا", "GradientTape: record forward, replay backward")
    definition("**الاشتقاق الآلي** `automatic differentiation`: أثناء التمرير الأمامي داخل `with tf.GradientTape() as tape:` يسجّل TensorFlow كل عملية ونتيجتها (رسم حسابي). "
               "ثم `tape.gradient(loss, variables)` يمشي الرسم **عكسيًا** ويطبق قاعدة السلسلة عند كل عقدة. لا اشتقاق رمزي ولا فروق عددية — تدرجات دقيقة بكلفة تمرير أمامي تقريبًا.")
    n = tape_numbers()
    tcaps = [
        f"**بدء التسجيل**: عصبون واحد بمدخلين `x = ({n['x'][0]}, {n['x'][1]})`، أوزان `w = ({n['w'][0]}, {n['w'][1]})` وانحياز `b = {n['b']}`، والحقيقة `y = {n['y']:.0f}`. المتغيرات (بنفسجي) هي ما سنشتق بالنسبة إليه.",
        f"**عملية 1 مسجلة**: `x1·w1 = {n['m1']:.2f}`. الشريط يحفظ المدخلين والنتيجة ليعرف لاحقًا أن ∂(x1·w1)/∂w1 = x1.",
        f"**عملية 2**: `x2·w2 = {n['m2']:.2f}`.",
        f"**عملية 3**: الجمع مع الانحياز: `z = {n['m1']:.2f} + ({n['m2']:.2f}) + {n['b']} = {n['z']:.3f}`.",
        f"**عملية 4**: `p = σ(z) = {n['p']:.4f}` — احتمال أن y = 1.",
        f"**عملية 5**: الخسارة `L = −log p = {n['L']:.4f}`. انتهى `with`؛ الشريط يحمل الرسم كله.",
        f"**`tape.gradient` يبدأ من النهاية**: ∂L/∂L = 1، ثم ∂L/∂p = −1/p = {n['dL_dp']:.3f}.",
        f"**عبر sigmoid**: ∂p/∂z = p(1−p) = {n['dp_dz']:.4f}، فـ ∂L/∂z = {n['dL_dp']:.3f} × {n['dp_dz']:.4f} = **{n['dL_dz']:.4f}** = p − y بالضبط (كما اشتققنا في الأسبوع 02).",
        f"**إلى المتغيرات**: عقدة الضرب تمرر ∂L/∂z مضروبًا في «الشريك»: ∂L/∂w1 = {n['dL_dz']:.4f} × {n['x'][0]} = {n['g']['w1']:.4f}، ∂L/∂w2 = × {n['x'][1]} = {n['g']['w2']:.4f}، ∂L/∂b = {n['g']['b']:.4f}.",
        f"**`apply_gradients`**: كل متغير يتحرك عكس تدرجه بخطوة η = {n['lr']}. التدرجات سالبة ⇒ الأوزان تزيد ⇒ z يكبر ⇒ p يقترب من 1 = الحقيقة.",
    ]
    stages = ["record", "record", "record", "record", "record", "record", "backward", "backward", "backward", "update"]
    cur = animation_player("w03_tape", [Frame(tape_svg(n, i), caption(c), action=stages[i], highlight=0 if i <= 5 else (1 if i < 9 else 2)) for i, c in enumerate(tcaps)],
                           title_ar="شريط التدرج على عصبون واحد", stages=["forward (recorded)", "tape.gradient (backward)", "apply_gradients"], interval_ms=2300)
    worked_steps([
        ("قاعدة السلسلة من الخسارة إلى z", rf"\frac{{\partial L}}{{\partial z}} = \frac{{\partial L}}{{\partial p}}\cdot\frac{{\partial p}}{{\partial z}} = ({n['dL_dp']:.3f})({n['dp_dz']:.4f}) = {n['dL_dz']:.4f}"),
        ("من z إلى الوزن الأول", rf"\frac{{\partial L}}{{\partial w_1}} = \frac{{\partial L}}{{\partial z}}\cdot x_1 = ({n['dL_dz']:.4f})({n['x'][0]}) = {n['g']['w1']:.4f}"),
        ("التحديث بالنزول بالتدرج", rf"w_1 \leftarrow {n['w'][0]} - {n['lr']}\times({n['g']['w1']:.4f}) = {n['new']['w1']:.4f}"),
    ], title_ar="نفس الأرقام بالرموز")
    math_note("كل عقدة تعرف مشتقتها المحلية فقط (الضرب: الشريك؛ الجمع: 1؛ sigmoid: p(1−p)). الاشتقاق الآلي = ضرب المشتقات المحلية على طول المسار — هذا هو الانتشار العكسي (الأسس 14) منفذًا آليًا لملايين العقد.")
    compare_table(["", "TensorFlow", "PyTorch"],
                  [("التسجيل", "`with tf.GradientTape() as tape:`", "تلقائي لكل موتر `requires_grad=True`"), ("التدرج", "`grads = tape.gradient(loss, vars)` (قيم مرجعة)", "`loss.backward()` (يملأ `p.grad`)"),
                   ("التحديث", "`opt.apply_gradients(zip(grads, vars))`", "`opt.step()`"), ("تصفير", "غير لازم (شريط جديد كل مرة)", "`opt.zero_grad()` لازم")],
                  ["rtl", "code", "code"])

    # ------------------------------------------------------------------ tf.data
    h2("tf.data: من أين تأتي الدفعات؟", "tf.data: where batches come from")
    states = shuffle_sim()
    scaps = [f"**البداية**: `shuffle(buffer_size=4)` يملأ مخزنًا بأول 4 عناصر (0–3) من المصدر المرتب. الباقي ينتظر."]
    for k, s in enumerate(states[1:], 1):
        scaps.append(f"**سحب {k}**: يُختار عنصر عشوائي من المخزن (**{s['pick']}**) ويُرسل للمخرج، ويحل محله العنصر التالي من المصدر."
                     + (" المصدر نفد؛ المخزن يُفرَّغ." if not s["src"] else ""))
    sframes = [Frame(shuffle_svg(s), caption(c), action="fill" if i == 0 else f"pick {s['pick']}") for i, (s, c) in enumerate(zip(states, scaps))]
    sframes.append(Frame(shuffle_svg(states[-1], final=True), caption("**`.batch(4)`** يجمع المخرج في دفعات من 4، و**`.prefetch`** يجهّز الدفعة التالية بالتوازي مع التدريب. "
                                                                   "لاحظ: المخزن الصغير يخلط **محليًا** فقط — العنصر 11 لا يمكن أن يظهر أولًا. لخلط جيد: `buffer_size` ≥ حجم البيانات (هنا 300 في الكود)."), action="batch + prefetch"))
    animation_player("w03_shuffle", sframes, title_ar="shuffle(4) → batch(4) → prefetch على 12 عنصرًا", interval_ms=1300)
    compare_table(["التحويل", "ماذا يفعل", "خطأ شائع"],
                  [("`from_tensor_slices((X, y))`", "كل صف = عنصر (x_i, y_i)", "تمرير DataFrame بأعمدة object"), ("`shuffle(buffer)`", "خلط بمخزن متدحرج", "مخزن صغير = خلط ضعيف؛ أو الخلط بعد batch (يخلط الدفعات لا العناصر)"),
                   ("`batch(32)`", "تجميع العناصر في موتر (32, …)", "ثم تمرير `batch_size=` لـ fit أيضًا"), ("`prefetch(AUTOTUNE)`", "تحضير الدفعة التالية أثناء التدريب", "لا شيء — أضفه دائمًا")],
                  ["code", "rtl", "rtl"])

    # ------------------------------------------------------------------ code lab
    h2("الآن بالكود: خطوة، حلقة، ثم fit على Dataset", "Now in code")
    code_lab(CodeLab(
        key="w03_hood", title_ar="خطوة fit يدويًا، ثم حلقة كاملة، ثم fit على Dataset", code=CODE, level="B",
        before=Before(goal_ar="رؤية ما يحدث في تحديث واحد (أشكال التدرجات، وزن قبل/بعد، خسارة قبل/بعد)، ثم ثلاث حقب بسطر سجل بشكل Keras، ثم fit على tf.data.", stage_ar="الأسبوع 03: تحت الغطاء.",
                      inputs_ar="moons (300 تدريب / 100 تحقق) كـ tf.data.Dataset.", expected_ar="دفعة (32, 2)/(32, 1) على CPU؛ 4 تدرجات بأشكال المعلمات؛ الوزن يتغير والخسارة تنخفض بعد الخطوة؛ ثلاثة أسطر سجل؛ fit(ds) يعمل.", prerequisites_ar="الوحدة 21: GradientTape، tf.data، الحلقة المخصصة."),
        explain=[("7", "خط بيانات كامل: from_tensor_slices → shuffle → batch → prefetch. هذا ما يفعله fit بالمصفوفات ضمنيًا."), ("14-22", "خطوة واحدة: دفعة → شريط → أمامي بـ training=True → خسارة → تدرج لكل معلمة → apply_gradients. الوزن يتغير والخسارة تنخفض على **نفس الدفعة**."),
                 ("25-33", "الحلقة كاملة: تصفير المقاييس، حلقة الدفعات، تجميع موزون، تحقق بـ training=False. السطر المطبوع = سطر fit."), ("36-38", "الغطاء مرة أخرى: compile + fit على Dataset (batch_size داخل الخط). الجهاز: [] هنا = CPU؛ في Colab مع GPU تُوضع الموترات عليه تلقائيًا بلا تغيير في الكود.")],
        run=run_printed(CODE),
        after_ar="- 4 تدرجات لأربع معلمات (kernel, bias × طبقتين) بنفس أشكالها: هذا هو `.grad` في PyTorch لكن كقيم مرجعة.\n- الخسارة بعد الخطوة أقل على نفس الدفعة — ليس ضمانًا للتعميم، لكنه دليل أن الاتجاه صحيح.\n- الأسطر الثلاثة تطابق شكل سجل Keras لأن المصدر واحد.",
    ))

    h3("الأجهزة: CPU وGPU", "Devices")
    compare_table(["السؤال", "الجواب في TensorFlow/Keras"],
                  [("أين يعمل الكود؟", "على GPU تلقائيًا إن وُجد، وإلا CPU — بلا تغيير في الكود."), ("كيف أتحقق؟", "`tf.config.list_physical_devices('GPU')` — قائمة فارغة = CPU فقط."),
                   ("أين الموتر؟", "`t.device` مثل `/job:localhost/.../device:CPU:0`."), ("متى يفيد GPU؟", "شبكات كبيرة ودفعات كبيرة (صور، تسلسلات). شبكتنا الصغيرة لا تستفيد تقريبًا — الأسبوع 13.")],
                  ["rtl", "rtl"])

    h2("أين اللقطات؟", "Where are the screens?")
    compare_table(["ما تريد رؤيته", "الصفحة"],
                  [("مخرج tf.Tensor، summary، سجل fit، evaluate/predict، رسم History، TensorBoard", "الوحدة 22: معرض TensorFlow وKeras"), ("موتر PyTorch، print(model)، سجل حلقة، CPU/GPU", "الوحدة 22: معرض PyTorch"), ("خطأ شكل وخطأ dtype/جهاز كما تظهر فعليًا", "الوحدة 22: معرض الأخطاء"), ("قبل/أثناء/بعد التدريب", "الوحدة 22: قبل/بعد"), ("لوحات TensorBoard مشروحة", "الوحدة 21: TensorBoard")],
                  ["rtl", "rtl"])
    with st.container(horizontal=True):
        st.button("المعرض — Keras/TensorFlow", icon=":material/photo_library:", on_click=go, args=("foundations.gallery.keras_tf",), key="w03_go_gal_k")
        st.button("المعرض — الأخطاء", icon=":material/bug_report:", on_click=go, args=("foundations.gallery.errors",), key="w03_go_gal_e")
        st.button("TensorBoard", icon=":material/monitoring:", on_click=go, args=("foundations.frameworks.tensorflow.tensorboard_errors",), key="w03_go_tb")
        st.button("معمل GradientTape", icon=":material/science:", on_click=go, args=("labs.gradient_tape_lab",), key="w03_lab_tape")
    intuition("Keras: fit تخفي الحلقة. TensorFlow: الحلقة بأدواته الأربع. PyTorch (الدرس التالي): الحلقة نفسها هي الأسلوب الافتراضي. ثلاث طبقات لنفس الرياضيات.")
    common_mistake("كتابة حلقة مخصصة في TensorFlow لمسألة قياسية «لأنها أوضح»: تخسر callbacks وHistory وأشرطة التقدم وتربح أخطاء. الحلقة المخصصة للحاجة غير القياسية؛ للتعلم اقرأها مرة كما هنا.")
    common_mistake("حساب الخسارة **خارج** `with tf.GradientTape()`: الشريط لم يسجّلها، فيعيد `tape.gradient` قيم `None` لكل المتغيرات والنموذج لا يتعلم بصمت.")
    quiz("w03.hood", [
        Q("`tape.gradient(loss, model.trainable_variables)` يعيد…", ["عددًا", "قائمة تدرجات بأشكال المعلمات", "الخسارة"], 1, ""),
        Q("`model(xb, training=True)` مقابل `training=False`:", ["لا فرق", "يفعّل Dropout/إحصاءات BN الدفعية", "يحدّث الأوزان"], 1, ""),
        Q("`fit(ds)` مع Dataset مقسّم بـ batch(32):", ["مرّر batch_size=32 أيضًا", "لا تمرّر batch_size", "يفشل"], 1, ""),
        Q("مع GPU في Colab، كود Keras…", ["يُعاد كتابته", "لا يتغير", "يحتاج .to(device)"], 1, ""),
        Q("في المثال: p = 0.49 وy = 1. ∂L/∂z =", ["0.49", "−0.51", "1"], 1, "p − y."),
        Q("عقدة الضرب z = x·w تمرر إلى w التدرج…", ["∂L/∂z", "∂L/∂z × x", "x"], 1, "المشتقة المحلية = الشريك."),
        Q("`shuffle(4)` على 1000 عنصر مرتب…", ["خلط كامل", "خلط محلي ضعيف", "لا خلط"], 1, "المخزن الصغير."),
        Q("`tape.gradient` أعاد None لكل المتغيرات. السبب الأرجح:", ["η صغير", "الخسارة حُسبت خارج كتلة with", "الدفعة كبيرة"], 1, ""),
    ])
    takeaway("fit = tf.data + GradientTape + apply_gradients + metrics في حلقتين. الشريط يسجّل أماميًا ويضرب المشتقات المحلية عكسيًا. shuffle يخلط بمخزن متدحرج. رأيتها مرة؛ استخدم fit في العمل.")
    lesson_footer(LESSON, ["GradientTape بأرقام دقيقة (تحريك).", "tf.data عنصرًا عنصرًا (تحريك).", "خطوة وحلقة وfit(ds) بالكود.", "الأجهزة وروابط اللقطات."])
