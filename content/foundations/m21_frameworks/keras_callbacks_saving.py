import streamlit as st

from components.callouts import common_mistake, definition, intuition, practical_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.frameworks.keras.callbacks_saving",
    title_ar="الاستدعاءات والحفظ: EarlyStopping وModelCheckpoint وجداول معدل التعلم وsave/load",
    title_en="Callbacks & Saving: EarlyStopping, ModelCheckpoint, LR Callbacks, save/load",
    module="foundations.frameworks",
    parent="foundations.frameworks.keras",
    order=8,
    prerequisites=["foundations.frameworks.keras.fit", "foundations.regularization.early_stopping", "foundations.optim.schedules_convergence"],
    objectives_ar=["فهم الاستدعاء كدالة تُستدعى عند أحداث الحلقة (بداية/نهاية حقبة أو دفعة).", "استخدام EarlyStopping وModelCheckpoint وReduceLROnPlateau وLearningRateScheduler بمعاملاتها.", "كتابة استدعاء مخصص بسيط، وحفظ/تحميل نموذج كامل وأوزان فقط."],
    terms=["epoch", "learning_rate"],
    difficulty="intermediate",
    summary_ar="Callback = كود يعمل عند أحداث fit. EarlyStopping (patience, restore_best_weights)، ModelCheckpoint (save_best_only)، ReduceLROnPlateau، LearningRateScheduler. الحفظ: model.save('m.keras') وkeras.saving.load_model.",
)

CODE = '''import os, tempfile; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, keras
from keras import layers, callbacks
keras.utils.set_random_seed(0)
rng = np.random.default_rng(0)
X = rng.normal(size=(400, 6)).astype("float32"); y = (np.sin(X[:, 0]) + X[:, 1] * X[:, 2] > 0).astype("float32")
X_tr, y_tr, X_va, y_va = X[:300], y[:300], X[300:], y[300:]

def build():
    m = keras.Sequential([layers.Input(shape=(6,)), layers.Dense(64, activation="relu"), layers.Dense(64, activation="relu"), layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=keras.optimizers.Adam(3e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m

class LRLogger(callbacks.Callback):                                  # استدعاء مخصص: سطر لكل حقبة
    def on_epoch_end(self, epoch, logs=None):
        lr = float(self.model.optimizer.learning_rate)
        if epoch % 10 == 0 or epoch < 3:
            print(f"  epoch {epoch+1:3d}  lr={lr:.5f}  loss={logs['loss']:.4f}  val_loss={logs['val_loss']:.4f}")

tmp = os.path.join(tempfile.mkdtemp(), "best.keras")
cbs = [
    callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0),   # وحدة 20
    callbacks.ModelCheckpoint(tmp, monitor="val_loss", save_best_only=True, verbose=0),             # يحفظ أفضل نموذج على القرص
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0),  # يخفض η عند الثبات
    LRLogger(),
]
model = build()
h = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=100, batch_size=32, verbose=0, callbacks=cbs)
n_run = len(h.history["loss"]); best = int(np.argmin(h.history["val_loss"])) + 1
print(f"stopped after {n_run} epochs (max 100); best val_loss {min(h.history['val_loss']):.4f} at epoch {best}; final lr {float(model.optimizer.learning_rate):.6f}")
print("restored best weights? val_loss now =", round(model.evaluate(X_va, y_va, verbose=0)[0], 4))

# الحفظ والتحميل: النموذج كاملًا (بنية + أوزان + compile)
model.save(os.path.join(os.path.dirname(tmp), "final.keras"))
loaded = keras.saving.load_model(os.path.join(os.path.dirname(tmp), "final.keras"))
print("loaded predictions identical?", np.allclose(loaded.predict(X_va[:5], verbose=0), model.predict(X_va[:5], verbose=0)))
# الأوزان فقط: تحتاج نفس البنية
w_path = os.path.join(os.path.dirname(tmp), "w.weights.h5"); model.save_weights(w_path)
fresh = build(); fresh.load_weights(w_path)
print("weights-only reload identical?", np.allclose(fresh.predict(X_va[:5], verbose=0), model.predict(X_va[:5], verbose=0)))
print("best checkpoint exists on disk?", os.path.exists(tmp), "| size KB:", round(os.path.getsize(tmp) / 1024, 1))'''

CUSTOM = '''class Callback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None): ...          # قبل الحلقة
    def on_epoch_begin(self, epoch, logs=None): ...
    def on_train_batch_end(self, batch, logs=None): ...  # logs = {"loss": …, "accuracy": …}
    def on_epoch_end(self, epoch, logs=None): ...     # logs تحوي val_* أيضًا
    def on_train_end(self, logs=None): ...
    # داخل أي منها: self.model (النموذج)، self.model.optimizer.learning_rate، self.model.stop_training = True'''


def render() -> None:
    lesson_header(LESSON)
    definition("**الاستدعاء** `Callback`: كائن يملك دوالًا تستدعيها `fit` عند أحداث محددة: بداية/نهاية التدريب، الحقبة، الدفعة. يصل إلى النموذج والمحسّن والسجلات (`logs`) ويمكنه تغييرها أو إيقاف التدريب. بهذا تُضاف إلى الحلقة المخفية سلوكيات كنت تكتبها يدويًا في الوحدة 16 (الإيقاف المبكر) دون فتحها.")
    why("لأن `fit` تخفي الحلقة، تحتاج **نقاط دخول** كي تقول: «إن لم يتحسن val_loss لـ 8 حقب أوقف واسترجع أفضل أوزان»، «احفظ أفضل نموذج»، «اخفض η عند الثبات». الاستدعاءات هي تلك النقاط. في PyTorch تكتب هذا مباشرة داخل حلقتك.")
    h2("الاستدعاءات الجاهزة", "Built-in callbacks")
    compare_table(["الاستدعاء", "ماذا يفعل", "المعاملات المهمة", "الوحدة"],
                  [("`EarlyStopping`", "يوقف التدريب عندما يتوقف مقياس المراقبة عن التحسن", "`monitor='val_loss'`, `patience`, `min_delta`, `restore_best_weights=True`, `mode`", "20"),
                   ("`ModelCheckpoint`", "يحفظ النموذج (أو الأوزان) على القرص عند كل حقبة أو عند التحسن", "`filepath`, `monitor`, `save_best_only=True`, `save_weights_only`", "—"),
                   ("`ReduceLROnPlateau`", "يضرب η في `factor` عندما يثبت المقياس `patience` حقب", "`factor=0.5`, `patience`, `min_lr`", "15"),
                   ("`LearningRateScheduler`", "η = f(epoch) بدالة تكتبها (تناقص أسي، خطوي…)", "`schedule(epoch, lr)`", "15"),
                   ("`TensorBoard`", "يكتب السجلات لأداة TensorBoard", "`log_dir`, `histogram_freq`", "درس TensorBoard"),
                   ("`CSVLogger`", "يكتب History إلى ملف CSV كل حقبة", "`filename`", "—"),
                   ("`TerminateOnNaN`", "يوقف التدريب عند خسارة NaN", "—", "14 (الانفجار)")],
                  ["code", "rtl", "code", "rtl"])
    code_lab(CodeLab(
        key="keras_callbacks", title_ar="أربعة استدعاءات معًا + حفظ/تحميل بطريقتين", code=CODE, level="B",
        before=Before(goal_ar="تدريب حتى 100 حقبة مع إيقاف مبكر واسترجاع أفضل أوزان، حفظ أفضل نموذج، خفض η عند الثبات، واستدعاء مخصص يطبع η — ثم حفظ النموذج كاملًا وأوزانه فقط والتحقق من تطابق التنبؤات بعد التحميل.", stage_ar="Keras: callbacks → save/load.",
                      inputs_ar="300 تدريب / 100 تحقق × 6 خصائص، هدف ثنائي.", expected_ar="توقف قبل 100 حقبة؛ أفضل حقبة < الأخيرة؛ val_loss بعد التدريب = الأفضل (استرجاع)؛ η نهائي أصغر من 3e-3؛ تنبؤات مطابقة بعد التحميل بالطريقتين."),
        explain=[("14-18", "استدعاء مخصص: يرث `Callback`، يعرّف `on_epoch_end`، يقرأ η من `self.model.optimizer`."), ("21-26", "قائمة الاستدعاءات. EarlyStopping بـ patience=8 واسترجاع الأفضل؛ ModelCheckpoint يحفظ فقط عند تحسن val_loss؛ ReduceLROnPlateau يخفض η ×0.5 بعد 3 حقب بلا تحسن."),
                 ("28-31", "epochs=100 حد أعلى. عدد الحقب الفعلي = len(history). `evaluate` بعد fit يعطي أفضل val_loss لأن الأوزان استُرجعت."), ("34-36", "الحفظ الكامل بصيغة `.keras`: بنية + أوزان + إعدادات compile. التحميل يعيد نموذجًا جاهزًا."), ("38-40", "أوزان فقط (`.weights.h5`): تحتاج بناء نفس البنية أولًا ثم `load_weights`.")],
        run=run_printed(CODE),
        after_ar="- الأسطر التي طبعها LRLogger تُظهر η يتناقص عند الثبات: هذا `ReduceLROnPlateau` يعمل.\n- «stopped after N epochs» مع N < 100: `EarlyStopping` عمل؛ `best` هو الحقبة التي حُفظت وأُرجعت.\n- التطابق بعد التحميل True بالطريقتين: الحفظ ليس ضياعًا للدقة.",
    ))
    h2("استدعاء مخصص: الهيكل", "Custom callback skeleton")
    st.code(CUSTOM, language="python")
    intuition("`logs` هو نفس القاموس الذي يملأ History: في `on_epoch_end` يحوي `loss`, `accuracy`, `val_loss`, `val_accuracy`. أي شيء تريد فعله «كل حقبة» (رسم، إرسال، فحص NaN، حفظ صورة) مكانه هنا.")
    h2("الحفظ والتحميل", "Saving & loading")
    compare_table(["الطريقة", "ماذا يُحفظ", "الاستعادة", "متى"],
                  [("`model.save('m.keras')`", "البنية + الأوزان + حالة compile (+ حالة المحسّن)", "`keras.saving.load_model('m.keras')` → نموذج جاهز", "الافتراضي: تسليم، نشر، استئناف تدريب"),
                   ("`model.save_weights('m.weights.h5')`", "الأوزان فقط", "ابنِ نفس البنية ثم `load_weights`", "نقل أوزان بين نسخ الكود؛ نقاط حفظ خفيفة"),
                   ("`model.export('dir')`", "رسم استدلال بصيغة SavedModel (TensorFlow)", "`tf.saved_model.load` / TF Serving", "الإنتاج خارج بايثون (تعميق)")],
                  ["code", "rtl", "code", "rtl"])
    practical_note("احفظ دائمًا مع النموذج: **النسخة** (`keras.__version__`)، **معلمات التحجيم** للبيانات (المتوسط/الانحراف من التدريب — وحدة 8)، و**العتبة** المختارة (وحدة 18). النموذج بلا هذه الثلاثة لا يمكن استخدامه بشكل صحيح.")
    common_mistake("`restore_best_weights=False` (الافتراضي!) مع EarlyStopping: يتوقف التدريب لكن الأوزان تبقى عند **آخر** حقبة (بعد patience حقب من التدهور) لا عند الأفضل. اضبطه True أو استخدم ModelCheckpoint(save_best_only=True) وحمّل الملف.")
    quiz("keras.callbacks", [
        Q("`patience=8` في EarlyStopping تعني…", ["8 حقب كحد أقصى", "التوقف بعد 8 حقب متتالية بلا تحسن", "التحقق كل 8 حقب"], 1, "صبر."),
        Q("`ReduceLROnPlateau(factor=0.5)`…", ["يضاعف η", "يضرب η في 0.5 عند الثبات", "يوقف التدريب"], 1, "خفض."),
        Q("`model.save('m.keras')` يحفظ…", ["الأوزان فقط", "البنية والأوزان وإعدادات compile", "History"], 1, "كامل."),
        Q("لتحميل أوزان فقط تحتاج…", ["لا شيء", "بناء نفس البنية أولًا", "إعادة التدريب"], 1, "الأشكال يجب أن تطابق."),
    ])
    takeaway("Callback = نقاط دخول إلى الحلقة المخفية. EarlyStopping(restore_best_weights=True) + ModelCheckpoint(save_best_only=True) + ReduceLROnPlateau = الثلاثي القياسي. save('.keras') كامل؛ save_weights أوزان فقط.")
    lesson_footer(LESSON, ["الاستدعاءات الجاهزة وأهم معاملاتها.", "هيكل الاستدعاء المخصص.", "الحفظ الكامل مقابل الأوزان."])
