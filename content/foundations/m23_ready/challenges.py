import numpy as np
import streamlit as st

from components.callouts import intuition, takeaway, why
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q
from content.foundations.m23_ready._scoring import record, scored_quiz, scores
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.ready.challenges",
    title_ar="التحديات الست: الأشكال، الحقب/الدفعات، الخسارة/المحسّن، Keras، TensorFlow، PyTorch",
    title_en="The Six Challenges: Shapes, Epoch/Batch, Loss/Optimizer, Keras, TensorFlow, PyTorch",
    module="foundations.ready",
    order=3,
    prerequisites=["foundations.ready.prerequisite_check"],
    objectives_ar=["حل ست تحديات عملية تُصحَّح آليًا وتُسجَّل لحالة الجاهزية.", "كل تحدٍّ يولّد مسألة جديدة بزر «مسألة أخرى» حتى تتقنه.", "الأخطاء تعيدك إلى الدرس المناسب."],
    terms=[],
    difficulty="intermediate",
    summary_ar="1) أشكال ومعلمات شبكة عشوائية. 2) دفعات وتحديثات. 3) اختيار الخسارة والتصرف مع η. 4) ترتيب سير عمل Keras وملء الفراغات. 5) تدرج يدوي مقابل GradientTape. 6) اصطياد الأخطاء في حلقة PyTorch.",
)


def _rng(name: str) -> np.random.Generator:
    st.session_state.setdefault("ready_seeds", {})
    seed = st.session_state["ready_seeds"].setdefault(name, 0)
    return np.random.default_rng(seed)


def _new(name: str) -> None:
    st.session_state["ready_seeds"][name] = st.session_state["ready_seeds"].get(name, 0) + 1
    for k in list(st.session_state.keys()):
        if k.startswith(f"ch_{name}_"):
            del st.session_state[k]


def _shape_challenge() -> None:
    h2("التحدي 1: أشكال الموترات والمعلمات", "Challenge 1: Tensor shapes & parameters")
    rng = _rng("shape")
    f = int(rng.choice([3, 5, 8, 12, 20])); h1 = int(rng.choice([4, 8, 16, 32])); h2_ = int(rng.choice([0, 4, 8, 16])); k = int(rng.choice([1, 3, 5, 10])); b = int(rng.choice([16, 32, 64]))
    layers = [("Input", f)] + [("Dense", h1)] + ([("Dense", h2_)] if h2_ else []) + [("Dense", k)]
    code = f"model = keras.Sequential([\n    layers.Input(shape=({f},)),\n    layers.Dense({h1}, activation='relu'),\n" + (f"    layers.Dense({h2_}, activation='relu'),\n" if h2_ else "") + f"    layers.Dense({k}, activation='{'sigmoid' if k == 1 else 'softmax'}'),\n])\nX_batch.shape == ({b}, {f})"
    st.code(code, language="python")
    st.markdown(f"دفعة من **{b}** ملاحظة تدخل الشبكة. أجب:")
    c1, c2, c3 = st.columns(3)
    with c1:
        a1 = st.text_input("شكل مخرج الطبقة الكثيفة الأولى (مثل 32,16)", key="ch_shape_a1")
    with c2:
        a2 = st.text_input("شكل مخرج النموذج", key="ch_shape_a2")
    with c3:
        a3 = st.number_input("عدد المعلمات الكلي", min_value=0, step=1, key="ch_shape_a3")
    params = 0; prev = f
    for _, u in layers[1:]:
        params += prev * u + u; prev = u
    if st.button("تحقق", key="ch_shape_check", type="primary"):
        def parse(s):
            try:
                return tuple(int(x) for x in s.replace("(", "").replace(")", "").replace(" ", "").split(",") if x)
            except ValueError:
                return None
        ok1 = parse(a1) == (b, h1); ok2 = parse(a2) == (b, k); ok3 = int(a3) == params
        n_ok = sum([ok1, ok2, ok3]); record("shape", n_ok, 3, "foundations.architecture")
        table(["السؤال", "إجابتك", "الصحيح", ""], [("مخرج الأولى", a1 or "—", f"({b}, {h1})", "✅" if ok1 else "❌"), ("مخرج النموذج", a2 or "—", f"({b}, {k})", "✅" if ok2 else "❌"), ("المعلمات", str(int(a3)), str(params), "✅" if ok3 else "❌")], ["rtl", "code", "code", "rtl"])
        detail = " + ".join(f"({p}×{u}+{u})" for p, u in zip([f] + [u for _, u in layers[1:-1]], [u for _, u in layers[1:]]))
        st.markdown(f"**الحساب:** {detail} = {params}. بُعد الدفعة ({b}) يمر كما هو؛ آخر بُعد = وحدات الطبقة. (الوحدتان 10 و12)")
    st.button("مسألة أخرى", key="ch_shape_new", on_click=_new, args=("shape",))


def _batch_challenge() -> None:
    h2("التحدي 2: الحقب والدفعات والتحديثات", "Challenge 2: Epochs, batches & updates")
    rng = _rng("batch")
    n = int(rng.choice([300, 480, 1000, 1250, 5000])); bs = int(rng.choice([16, 32, 64, 100, 128])); ep = int(rng.choice([3, 5, 10, 20]))
    st.code(f"model.fit(X_train, y_train, epochs={ep}, batch_size={bs})   # X_train.shape == ({n}, 8)", language="python")
    c1, c2, c3 = st.columns(3)
    with c1:
        a1 = st.number_input("دفعات (تحديثات) لكل حقبة", min_value=0, step=1, key="ch_batch_a1")
    with c2:
        a2 = st.number_input("إجمالي التحديثات", min_value=0, step=1, key="ch_batch_a2")
    with c3:
        a3 = st.number_input("حجم آخر دفعة في الحقبة", min_value=0, step=1, key="ch_batch_a3")
    spe = int(np.ceil(n / bs)); last = n - (spe - 1) * bs
    if st.button("تحقق", key="ch_batch_check", type="primary"):
        ok = [int(a1) == spe, int(a2) == spe * ep, int(a3) == last]; record("batch", sum(ok), 3, "foundations.batch_epoch")
        table(["السؤال", "إجابتك", "الصحيح", ""], [("دفعات/حقبة", str(int(a1)), f"⌈{n}/{bs}⌉ = {spe}", "✅" if ok[0] else "❌"), ("إجمالي التحديثات", str(int(a2)), f"{spe} × {ep} = {spe * ep}", "✅" if ok[1] else "❌"), ("آخر دفعة", str(int(a3)), f"{n} − {spe - 1}×{bs} = {last}", "✅" if ok[2] else "❌")], ["rtl", "num", "code", "rtl"])
        st.markdown("سطر السجل سيعرض `" + f"{spe}/{spe}" + "` في كل حقبة. (الوحدة 17)")
    st.button("مسألة أخرى", key="ch_batch_new", on_click=_new, args=("batch",))


LOSS_QS = [
    Q("تصنيف 4 فئات متنافية، الهدف أعداد صحيحة 0..3، المخرج Dense(4, softmax). الخسارة:", ["binary_crossentropy", "categorical_crossentropy", "sparse_categorical_crossentropy", "mse"], 2, "الوحدة 13: sparse للأعداد الصحيحة."),
    Q("انحدار على سعر عقار، المخرج Dense(1) بلا تنشيط، قيم متطرفة قليلة. الخسارة الأنسب:", ["binary_crossentropy", "huber أو mae", "categorical_crossentropy", "hinge"], 1, "الوحدة 13: متانة للقيم المتطرفة."),
    Q("ثنائي، الهدف 0/1 بعمود واحد، المخرج Dense(1, sigmoid):", ["binary_crossentropy", "sparse_categorical_crossentropy", "mse", "kl_divergence"], 0, "الوحدة 13."),
    Q("PyTorch: مخرج logits (n, 3)، هدف (n,) int64:", ["nn.BCEWithLogitsLoss", "nn.CrossEntropyLoss", "nn.MSELoss", "nn.BCELoss"], 1, "الوحدة 21."),
    Q("الخسارة تتذبذب بعنف وأحيانًا nan من الحقبة الأولى:", ["زد η", "قلّل η (وربما قصّ التدرج)", "زد batch_size فقط", "أضف طبقات"], 1, "الوحدة 15."),
    Q("الخسارة تنخفض ببطء شديد وبثبات على 100 حقبة، لا تذبذب:", ["قلّل η", "زد η أو استخدم Adam", "أوقف التدريب", "قلّل البيانات"], 1, "الوحدة 15."),
    Q("val_loss يثبت ثم يرتفع بينما loss ينخفض:", ["زد الحقب", "إيقاف مبكر / تنظيم / بيانات أكثر", "زد η", "احذف التحقق"], 1, "الوحدتان 19–20."),
    Q("Adam مقابل SGD بلا زخم على مسألة جديدة، بلا وقت لضبط η:", ["SGD أكثر أمانًا", "Adam افتراضي أكثر أمانًا", "لا فرق", "RMSprop فقط"], 1, "الوحدة 15."),
]

KERAS_STEPS = ["Build the model (layers)", "compile(optimizer, loss, metrics)", "fit(X_train, y_train, validation_data, epochs, batch_size, callbacks)", "evaluate(X_test, y_test)", "predict(X_new)", "save('model.keras')"]
KERAS_FILL = [
    Q("`layers.Input(shape=(___,))` لبيانات X بشكل (2000, 15):", ["2000", "15", "(2000, 15)", "1"], 1, "شكل الملاحظة الواحدة."),
    Q("`model.compile(optimizer=___, loss='sparse_categorical_crossentropy', metrics=['accuracy'])` بمعدل تعلم صريح 3e-4:", ["'adam'", "keras.optimizers.Adam(learning_rate=3e-4)", "3e-4", "Adam"], 1, "الكائن لمعلمة صريحة."),
    Q("للتوقف عند توقف تحسن val_loss واسترجاع أفضل أوزان:", ["callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]", "epochs=10", "shuffle=False", "verbose=0"], 0, "الوحدة 21: callbacks."),
    Q("بعد `history = model.fit(...)`، خسارة التحقق للحقبة الأخيرة:", ["history.val_loss", "history.history['val_loss'][-1]", "history[-1]", "model.val_loss"], 1, "قاموس قوائم."),
    Q("`model.predict(X_test)` لـ Dense(3, softmax) يعيد شكل:", ["(n,)", "(n, 3)", "(3,)", "(n, 1)"], 1, "n × units."),
]

TORCH_LOOP = '''for epoch in range(epochs):
    model.eval()                                   # A
    for xb, yb in train_loader:
        out = model(xb)                            # B
        loss = loss_fn(out, yb)                    # C
        loss.backward()                            # D
        optimizer.step()                           # E
        running += loss                            # F
    with torch.no_grad():                          # G
        val_loss = loss_fn(model(X_val), y_val)    # H'''
TORCH_BUGS = {"A": "يجب `model.train()` في بداية الحقبة (eval يعطّل Dropout/BN أثناء التدريب).", "D": "لا `optimizer.zero_grad()` قبل backward → تراكم التدرجات.", "F": "`running += loss` يبقي الرسم في الذاكرة؛ يجب `loss.item()`.", "G": "قبل التحقق يجب `model.eval()` (ثم العودة إلى train في الحقبة التالية)."}


def _torch_loop_challenge() -> None:
    h2("التحدي 6: اصطياد الأخطاء في حلقة PyTorch", "Challenge 6: PyTorch training-loop bug hunt")
    st.markdown("الحلقة أدناه تعمل بلا رسالة خطأ لكنها تحوي **أربعة** أخطاء صامتة. حدد الأسطر المعطوبة (بحرفها):")
    st.code(TORCH_LOOP, language="python")
    picked = st.multiselect("الأسطر المعطوبة", list("ABCDEFGH"), key="ch_torch_pick")
    if st.button("تحقق", key="ch_torch_check", type="primary"):
        correct = set(TORCH_BUGS); chosen = set(picked)
        tp = len(correct & chosen); fp = len(chosen - correct); fn = len(correct - chosen)
        score = max(0, tp - fp); record("torch_loop", score, 4, "foundations.frameworks")
        st.markdown(f"**صحيح: {tp} من 4 · خاطئ: {fp} · فائت: {fn}**")
        for k in "ABCDEFGH":
            if k in correct:
                st.markdown(f"- **{k}** — {'✅' if k in chosen else '❌ فائت'}: {TORCH_BUGS[k]}")
            elif k in chosen:
                st.markdown(f"- **{k}** — ❌ ليس خطأ: هذا السطر صحيح.")
        st.markdown("الحلقة الصحيحة في درس «حلقة التدريب في PyTorch» بالوحدة 21.")


def _tf_challenge() -> None:
    h2("التحدي 5: موترات TensorFlow والاشتقاق التلقائي", "Challenge 5: TensorFlow tensors & autodiff")
    rng = _rng("tf")
    a, c, w0, rows = int(rng.choice([2, 3, 4])), int(rng.choice([1, 2, 5])), float(rng.choice([0.5, 1.0, 2.0, 3.0])), int(rng.choice([8, 16, 32]))
    st.markdown("**(أ) الشكل**: ما شكل `z`؟")
    st.code(f"x = tf.random.normal(({rows}, {a}))\nW = tf.Variable(tf.random.normal(({a}, {c})))\nz = tf.matmul(x, W) + tf.zeros(({c},))", language="python")
    shp = st.text_input("شكل z (مثل 8,2)", key="ch_tf_shape")
    st.markdown("**(ب) التدرج**: احسب يدويًا ثم قارن:")
    st.code(f"w = tf.Variable({w0})\nwith tf.GradientTape() as tape:\n    L = ({a} * w - {c}) ** 2\ng = tape.gradient(L, w)   # = ?", language="python")
    gval = st.number_input("dL/dw", value=0.0, step=0.5, format="%.2f", key="ch_tf_g")
    if st.button("تحقق", key="ch_tf_check", type="primary"):
        from labs.fw import tf as _tf
        T = _tf()
        true_g = float(2 * (a * w0 - c) * a)
        w = T.Variable(w0)
        with T.GradientTape() as tape:
            L = (a * w - c) ** 2
        g_tf = float(tape.gradient(L, w))
        try:
            user_shape = tuple(int(x) for x in shp.replace("(", "").replace(")", "").replace(" ", "").split(",") if x)
        except ValueError:
            user_shape = None
        ok1 = user_shape == (rows, c); ok2 = abs(float(gval) - true_g) < 1e-6
        record("tf", int(ok1) + int(ok2), 2, "foundations.frameworks")
        table(["السؤال", "إجابتك", "الصحيح", ""], [("شكل z", shp or "—", f"({rows}, {c})  ← ({rows}, {a}) @ ({a}, {c}) + بث ({c},)", "✅" if ok1 else "❌"), ("dL/dw", f"{float(gval):.2f}", f"2({a}·{w0:g} − {c})·{a} = {true_g:g}  (GradientTape: {g_tf:g})", "✅" if ok2 else "❌")], ["rtl", "code", "code", "rtl"])
    st.button("مسألة أخرى", key="ch_tf_new", on_click=_new, args=("tf",))


def _keras_workflow_challenge() -> None:
    h2("التحدي 4: سير عمل Keras", "Challenge 4: Keras workflow")
    st.markdown("**(أ) رتّب الخطوات**: اختر الخطوة الصحيحة لكل موضع.")
    shuffled = list(KERAS_STEPS); np.random.default_rng(7).shuffle(shuffled)
    cols = st.columns(3)
    choice = []
    for i in range(6):
        with cols[i % 3]:
            choice.append(st.selectbox(f"الخطوة {i + 1}", ["—"] + shuffled, key=f"ch_keras_o{i}"))
    if st.button("تحقق من الترتيب", key="ch_keras_check", type="primary"):
        ok = sum(1 for i in range(6) if choice[i] == KERAS_STEPS[i]); record("keras_order", ok, 6, "foundations.frameworks")
        st.markdown(f"**{ok} / 6 في موضعها الصحيح.** الترتيب: " + " → ".join(f"`{s.split('(')[0].strip()}`" for s in KERAS_STEPS))
    st.markdown("**(ب) املأ الفراغات**")
    scored_quiz("keras_fill", "ready.keras_fill", KERAS_FILL, "foundations.frameworks", title_ar="ملء الفراغات")


def render() -> None:
    lesson_header(LESSON)
    why("الأسئلة الاختيارية تقيس التعرّف؛ التحديات تقيس **الإنتاج**: أن تحسب الشكل، وتعدّ التحديثات، وتختار الخسارة، وترتّب سير العمل، وتشتق بيدك، وتصطاد الخطأ الصامت. هذه هي المهارات التي يحتاجها الأسبوع الأول من المقرر الرسمي.")
    _shape_challenge()
    _batch_challenge()
    h2("التحدي 3: الخسارة والمحسّن", "Challenge 3: Loss & optimizer")
    scored_quiz("loss_optim", "ready.loss_optim", LOSS_QS, "foundations.loss", title_ar="اختر الخسارة والتصرّف الصحيح")
    _keras_workflow_challenge()
    _tf_challenge()
    _torch_loop_challenge()
    h2("نتائجك حتى الآن", "Your results so far")
    sc = scores()
    names = [("shape", "الأشكال"), ("batch", "الحقب/الدفعات"), ("loss_optim", "الخسارة/المحسّن"), ("keras_order", "ترتيب Keras"), ("keras_fill", "فراغات Keras"), ("tf", "TensorFlow"), ("torch_loop", "حلقة PyTorch")]
    rows = [(ar, f"{sc[k]['correct']} / {sc[k]['total']}" if k in sc else "لم يُحل") for k, ar in names]
    table(["التحدي", "النتيجة"], rows, ["rtl", "num"])
    intuition("حُلّ كل تحدٍّ مرتين على الأقل بمسألة جديدة: النجاح مرة قد يكون حظًا، ومرتين بأرقام مختلفة إتقان.")
    takeaway("ست تحديات إنتاجية مُصحَّحة ومُسجَّلة. النتائج تدخل في حالة الجاهزية في الصفحة التالية.")
    lesson_footer(LESSON, ["أشكال ومعلمات.", "دفعات وتحديثات.", "خسارة ومحسّن، Keras، TensorFlow، PyTorch."])
