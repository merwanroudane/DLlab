import numpy as np
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, definition, intuition, practical_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import pipeline
from labs.fw import keras_fit_trace

LESSON = Lesson(
    id="foundations.frameworks.keras.fit",
    title_ar="fit(): الحلقة المخفية — epochs وbatch_size والتحقق والخلط وHistory",
    title_en="fit(): The Hidden Loop — epochs, batch_size, validation, shuffle & History",
    module="foundations.frameworks",
    parent="foundations.frameworks.keras",
    order=6,
    prerequisites=["foundations.frameworks.keras.compile", "foundations.training_loop.the_loop", "foundations.batch_epoch.definitions"],
    objectives_ar=["ربط كل معامل في fit بخطوة في حلقة التدريب اليدوية: epochs، batch_size، validation_split/data، shuffle، verbose، callbacks.", "مشاهدة ما يحدث داخل fit دفعةً دفعة عبر محرك مرئي بتتبع حقيقي.", "قراءة كائن History وقاموسه."],
    terms=["epoch", "batch_size", "iteration", "batch"],
    labs=["labs.training_loop_simulator"],
    difficulty="intermediate",
    summary_ar="fit(X, y, epochs, batch_size, validation_data, shuffle, callbacks, verbose) = حلقة الوحدة 16 كاملة. يعيد History قاموسه history.history: قائمة لكل مقياس بطول epochs.",
)

STAGES = ["Arrays", "Batches", "Epoch", "Batch", "Forward", "Loss", "Autodiff", "Update", "Metrics", "Validation", "History"]

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, keras
from keras import layers
keras.utils.set_random_seed(0)
rng = np.random.default_rng(0)
X = rng.normal(size=(500, 3)).astype("float32"); y = (X[:, 0] * X[:, 1] + 0.3 * X[:, 2] > 0).astype("float32")

model = keras.Sequential([layers.Input(shape=(3,)), layers.Dense(16, activation="relu"), layers.Dense(1, activation="sigmoid")])
model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    X, y,
    epochs=5,               # مرور كامل على بيانات التدريب × 5
    batch_size=50,          # 400 تدريب / 50 = 8 تحديثات لكل حقبة
    validation_split=0.2,   # آخر 20% (100 ملاحظة) تُفصل قبل الخلط ولا تُدرَّب
    shuffle=True,           # خلط ترتيب التدريب في كل حقبة (لا يمس التحقق)
    verbose=0,              # 0 صامت، 1 شريط تقدم، 2 سطر لكل حقبة
)
print("type            :", type(history).__name__)
print("keys            :", list(history.history.keys()))
for k, v in history.history.items():
    print(f"{k:<13}: {np.round(v, 4)}")
print("epochs run      :", len(history.epoch), "| params trained on:", history.params)
print("best val_loss at epoch", int(np.argmin(history.history["val_loss"])) + 1)'''


def render() -> None:
    lesson_header(LESSON)
    definition("**`fit()`**: ينفذ حلقة التدريب كاملة — تقسيم المصفوفات إلى دفعات، حلقة الحقب، حلقة الدفعات، تمرير أمامي، خسارة، اشتقاق تلقائي، تحديث المحسّن، تحديث المقاييس، ثم تحقق في نهاية كل حقبة — ويعيد كائن **`History`**. كل معامل فيه يقابل سطرًا كتبته يدويًا في الوحدة 16.")
    why("«fit سحر» هو أخطر اعتقاد في هذه الوحدة. عندما تتوقف الخسارة أو تنفجر أو يختلف val_loss عن loss، تحتاج أن تعرف **أي خطوة داخل fit** مسؤولة. المحرك أدناه يفتح الصندوق بتتبع حقيقي من Keras (عبر Callback) لا بمحاكاة.")
    h2("محرك fit() المرئي", "fit() deep visualizer")
    pipeline(["model.fit(...)", "Dataset / Arrays", "Batch creation", "Epoch loop", "Batch loop", "Forward", "Loss", "Autodiff", "Optimizer update", "Metric update", "Validation", "History"])
    c1, c2, c3 = st.columns(3)
    with c1:
        epochs = st.slider("epochs", 1, 3, 2, key="kf_epochs")
    with c2:
        bs = st.select_slider("batch_size", options=[20, 40, 60, 120], value=40, key="kf_bs")
    with c3:
        lr = st.select_slider("learning_rate (SGD)", options=[0.1, 0.5, 1.0], value=0.5, key="kf_lr")
    tr = keras_fit_trace(8, int(epochs), int(bs), float(lr), 0)
    spe = tr["steps_per_epoch"]
    frames = [Frame("", caption(f"**`model.fit(X, y, epochs={epochs}, batch_size={bs}, validation_data=(X_val, y_val))`** — المدخل مصفوفات: X_train بشكل ({tr['n']}, 2)، y_train ({tr['n']},)، والتحقق {tr['n_val']} ملاحظة تُحفظ جانبًا."), action="Arrays", highlight=0),
              Frame("", caption(f"**إنشاء الدفعات**: {tr['n']} / {bs} = **{spe}** دفعة لكل حقبة (آخرها قد تكون أصغر). كل دفعة `X_batch.shape = ({bs}, 2)`. مع `shuffle=True` يُخلط الترتيب في بداية كل حقبة."), action="Batches", values=[("steps_per_epoch", "", str(spe))], highlight=1)]
    for e in tr["events"]:
        if e["kind"] == "epoch_begin":
            frames.append(Frame("", caption(f"**حلقة الحقب — الحقبة {e['epoch']}/{epochs}** تبدأ: خلط الفهارس، تصفير مجمّعات المقاييس."), action=f"Epoch {e['epoch']}", highlight=2))
        elif e["kind"] == "batch":
            b = e["batch"]
            frames.append(Frame("", caption(f"**حلقة الدفعات — الدفعة {b}/{spe}**: تُحمَّل `X_batch ({bs}, 2)` و`y_batch ({bs},)`."), action=f"Batch {b}", highlight=3))
            frames.append(Frame("", caption(f"الدفعة {b} — **أمامي**: `Dense(8, relu)` ثم `Dense(1, sigmoid)` → `y_hat ({bs}, 1)`. (وحدة 12)"), action="Forward", highlight=4))
            frames.append(Frame("", caption(f"الدفعة {b} — **الخسارة** binary_crossentropy على الدفعة = **{e['loss']:.4f}** (متوسط تراكمي للحقبة كما يعرضه Keras)."), action="Loss", values=[("loss", "", f"{e['loss']:.4f}")], highlight=5))
            frames.append(Frame("", caption(f"الدفعة {b} — **الاشتقاق التلقائي**: TensorFlow يسجّل العمليات ويحسب ∂L/∂θ لكل معلمة بقاعدة السلسلة (وحدة 14). لا تراه في الكود — هو داخل fit."), action="Autodiff", highlight=6))
            frames.append(Frame("", caption(f"الدفعة {b} — **تحديث المحسّن** (SGD, η={lr}): `W1[0,0]`: {e['w_before']:+.4f} → {e['w_after']:+.4f}."), action="Update", equation=f"W1[0,0] ← {e['w_before']:+.4f} − {lr}·g = {e['w_after']:+.4f}", values=[("W1[0,0]", f"{e['w_before']:+.4f}", f"{e['w_after']:+.4f}")], highlight=7))
            frames.append(Frame("", caption(f"الدفعة {b} — **تحديث المقاييس**: accuracy التراكمية للحقبة = **{e['acc']:.3f}**. ثم الدفعة التالية."), action="Metrics", values=[("accuracy", "", f"{e['acc']:.3f}")], highlight=8))
        else:
            frames.append(Frame("", caption(f"**نهاية الحقبة {e['epoch']} — التحقق**: تمرير أمامي فقط على {tr['n_val']} ملاحظة (بلا تدرج ولا تحديث): `val_loss = {e['val_loss']:.4f}`، `val_accuracy = {e['val_acc']:.3f}`."), action="Validation", values=[("val_loss", "", f"{e['val_loss']:.4f}"), ("val_accuracy", "", f"{e['val_acc']:.3f}")], highlight=9))
            frames.append(Frame("", caption(f"**History** تُلحق الحقبة {e['epoch']}: `loss={e['loss']:.4f}`, `accuracy={e['acc']:.3f}`, `val_loss={e['val_loss']:.4f}`, `val_accuracy={e['val_acc']:.3f}` — سطر السجل الذي تراه مع verbose=2."), action="History", highlight=10))
    animation_player("kf_anim", frames, title_ar="ماذا يحدث داخل model.fit()؟ (تتبع حقيقي عبر Callback)", stages=STAGES, interval_ms=1200)
    with st.expander("سجل fit الحقيقي لهذا التشغيل (verbose=2)", icon=":material/terminal:"):
        st.code(tr["log"].strip(), language="text")
    intuition(f"عدد التحديثات الحقيقية = {spe} × {epochs} = **{spe * epochs}**. سطر السجل الواحد لكل حقبة يخفي {spe} دورات كاملة من (أمامي، خسارة، خلفي، تحديث). ما يظهر في السجل هو **المتوسط التراكمي** على دفعات الحقبة لا قيمة آخر دفعة.")
    h2("معاملات fit سطرًا سطرًا", "fit arguments, line by line")
    compare_table(["المعامل", "المعنى", "المقابل اليدوي (وحدة 16)", "ملاحظات"],
                  [("`x, y`", "مصفوفات التدريب (أو tf.data.Dataset)", "`X_train, y_train`", "أشكالها تحدد كل شيء"),
                   ("`epochs`", "عدد المرورات الكاملة", "`for epoch in range(E)`", "معلمة فائقة؛ مع EarlyStopping تصبح حدًا أعلى"),
                   ("`batch_size`", "حجم الدفعة (افتراضي 32)", "`for s in range(0, n, B)`", "يحدد عدد التحديثات لكل حقبة = ⌈n/B⌉"),
                   ("`validation_split`", "كسر من **آخر** بيانات التدريب يُفصل قبل الخلط", "`X[:k], X[k:]`", "خطر إن كانت البيانات مرتّبة — اخلط قبله أو استخدم validation_data"),
                   ("`validation_data`", "مجموعة تحقق جاهزة (X_val, y_val)", "`X_val, y_val`", "مفضَّل: تحكم كامل بالتقسيم (وحدة 8)"),
                   ("`shuffle`", "خلط ترتيب التدريب كل حقبة (افتراضي True)", "`rng.permutation(n)`", "لا يخلط التحقق؛ أوقفه للمتسلسلات الزمنية المرتبة"),
                   ("`callbacks`", "كائنات تُستدعى عند أحداث الحلقة", "`if val_loss < best: …`", "EarlyStopping، ModelCheckpoint… (الصفحة التالية)"),
                   ("`verbose`", "0 صامت / 1 شريط / 2 سطر لكل حقبة", "`print(...)`", "2 في السجلات والملفات؛ 1 تفاعليًا"),
                   ("`class_weight`", "وزن لكل فئة في الخسارة", "خسارة موزونة", "لعدم التوازن (وحدة 8)")],
                  ["code", "rtl", "code", "rtl"])
    code_lab(CodeLab(
        key="keras_fit_history", title_ar="fit بمعاملاته الكاملة وقراءة History", code=CODE, level="B",
        before=Before(goal_ar="تشغيل fit بمعاملات صريحة ثم فحص كائن History: نوعه، مفاتيحه، أطوال القوائم، وأفضل حقبة.", stage_ar="Keras: fit → History.",
                      inputs_ar="500 ملاحظة × 3 خصائص، هدف ثنائي.", expected_ar="مفاتيح loss/accuracy/val_loss/val_accuracy، كل قائمة بطول 5، وحقبة أفضل val_loss."),
        explain=[("11-19", "كل معامل مع تعليقه. لاحظ: 500 × 0.8 = 400 للتدريب → 8 دفعات من 50."), ("20-23", "History كائن؛ `history.history` قاموس {اسم: قائمة بطول epochs}."), ("24-25", "`history.epoch` قائمة الحقب و`history.params` ما استُخدم. أفضل حقبة = argmin(val_loss) + 1.")],
        run=run_printed(CODE),
        after_ar="- المفاتيح بادئة `val_` لكل مقياس عندما يوجد تحقق.\n- القوائم بطول epochs (لا بطول التحديثات): قيمة واحدة لكل حقبة = المتوسط التراكمي على دفعاتها.\n- هذا القاموس هو ما ترسمه منحنيات التعلم (وحدة 19).",
    ))
    practical_note("`validation_split=0.2` يأخذ **آخر 20%** من المصفوفة كما هي، قبل أي خلط. إن كانت بياناتك مرتبة بالفئة أو بالزمن فالتحقق سيكون منحازًا (وحدة 8: التسريب/التقسيم). اخلط بنفسك أولًا أو مرّر `validation_data`.")
    common_mistake("«الخسارة في السجل هي خسارة آخر دفعة». لا: هي المتوسط التراكمي على دفعات الحقبة الحالية — لهذا تكون `loss` أحيانًا أعلى من `val_loss` في الحقب الأولى (الأوزان تحسنت أثناء الحقبة، والتحقق يُقاس بعدها كاملة).")
    quiz("keras.fit", [
        Q("n=400، batch_size=50، epochs=5: عدد التحديثات", ["5", "8", "40"], 2, "8 × 5."),
        Q("`validation_split=0.2`…", ["يخلط ثم يأخذ 20%", "يأخذ آخر 20% قبل الخلط", "يأخذ 20% عشوائيًا كل حقبة"], 1, "آخر جزء."),
        Q("`history.history['loss']` طولها", ["عدد الدفعات", "عدد الحقب", "عدد الملاحظات"], 1, "قيمة لكل حقبة."),
        Q("داخل fit، الاشتقاق التلقائي يقابل…", ["الوحدة 12", "الوحدة 14 (الانتشار الخلفي)", "الوحدة 18"], 1, "قاعدة السلسلة آليًا."),
    ])
    takeaway("fit = الحلقة كاملة: دفعات ← حقب ← (أمامي، خسارة، اشتقاق، تحديث، مقاييس) ← تحقق ← History. كل معامل يقابل سطرًا يدويًا. History قاموس قوائم بطول epochs.")
    lesson_footer(LESSON, ["المحرك: 11 مرحلة بتتبع حقيقي.", "جدول المعاملات.", "History وقراءته."])
