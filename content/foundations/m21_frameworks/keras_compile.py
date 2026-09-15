import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.frameworks.keras.compile",
    title_ar="compile(): المحسّن والخسارة والمقاييس",
    title_en="compile(): Optimizer, Loss & Metrics",
    module="foundations.frameworks",
    parent="foundations.frameworks.keras",
    order=5,
    prerequisites=["foundations.frameworks.keras.layers_models", "foundations.loss.loss_activation_compatibility", "foundations.optim.rmsprop_adam"],
    objectives_ar=["فهم أن compile يربط ثلاثة كائنات بالنموذج ولا يحسب شيئًا.", "اختيار المحسّن والخسارة والمقياس بحسب المسألة (وليس تقليدًا).", "معرفة فرق النص (`'mse'`) عن الكائن (`keras.losses.MeanSquaredError()`) ومتى تحتاج الكائن."],
    terms=["optimizer", "loss", "learning_rate", "cross_entropy"],
    difficulty="beginner",
    summary_ar="compile(optimizer, loss, metrics): المحسّن = قاعدة التحديث، الخسارة = ما يُشتق، المقاييس = ما يُراقب. لا تدريب هنا. توافق الخسارة مع التنشيط الأخير وترميز الهدف من الوحدة 13.",
)

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, keras
from keras import layers
keras.utils.set_random_seed(0)
rng = np.random.default_rng(0)
X = rng.normal(size=(300, 4)).astype("float32")
y_int = (X[:, 0] + X[:, 1] ** 2 > 0.5).astype("int32") + (X[:, 2] > 1).astype("int32")   # 3 فئات كأعداد صحيحة 0/1/2

model = keras.Sequential([layers.Input(shape=(4,)), layers.Dense(16, activation="relu"), layers.Dense(3, activation="softmax")])

# قبل compile: لا محسّن ولا خسارة
print("compiled?", model.compiled)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),        # قاعدة التحديث (وحدة 15)
    loss="sparse_categorical_crossentropy",                    # هدف أعداد صحيحة + softmax (وحدة 13)
    metrics=["accuracy"],                                      # للمراقبة فقط؛ لا يُشتق
)
print("compiled?", model.compiled, "| optimizer:", model.optimizer.__class__.__name__, "lr =", float(model.optimizer.learning_rate))
print("loss object:", model.loss)
# الأوزان لم تتغير بعد compile
w0 = model.layers[0].get_weights()[0].copy()
h = model.fit(X, y_int, epochs=3, batch_size=32, verbose=0)
w1 = model.layers[0].get_weights()[0]
print("weights changed by compile? No. by fit?", not np.allclose(w0, w1))
print("history keys:", list(h.history.keys()), "-> loss:", np.round(h.history["loss"], 3), "acc:", np.round(h.history["accuracy"], 3))

# إعادة compile بمحسّن آخر لا تفقد الأوزان
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.05, momentum=0.9), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("after recompile, weights kept?", np.allclose(w1, model.layers[0].get_weights()[0]))'''


def render() -> None:
    lesson_header(LESSON)
    definition("**`compile()`**: يربط بالنموذج ثلاثة كائنات: **المحسّن** (كيف تُحدَّث المعلمات)، **الخسارة** (الدالة التي يُحسب تدرجها)، **المقاييس** (ما يُبلَّغ عنه). لا يحسب ولا يدرّب ولا يغيّر الأوزان. بعده فقط يمكن `fit` و`evaluate`.")
    why("لماذا مرحلة منفصلة؟ لأن النموذج (الطبقات) مستقل عن **كيف** تدرّبه. نفس النموذج يمكن تدريبه بـ SGD ثم Adam، بخسارة ثم أخرى، دون إعادة بنائه. الفصل يعكس فصلًا مفاهيميًا تعلمته: البنية (وحدة 10) ≠ الهدف (وحدة 13) ≠ الخوارزمية (وحدة 15).")
    h2("المعاملات الثلاثة", "The three arguments")
    compare_table(["المعامل", "ماذا يقرر", "الخيارات الشائعة", "كيف تختار"],
                  [("`optimizer`", "قاعدة التحديث θ ← θ − η·(…)", "`'adam'`, `Adam(learning_rate=1e-3)`, `SGD(lr, momentum=0.9)`, `RMSprop`", "Adam افتراضي آمن؛ SGD+momentum عند الحاجة لتعميم أفضل مع جدول. معدل التعلم أهم معلمة فائقة (وحدة 15)."),
                   ("`loss`", "ما يُشتق وما يُقلَّل", "`'mse'`, `'binary_crossentropy'`, `'categorical_crossentropy'`, `'sparse_categorical_crossentropy'`", "بحسب نوع المهمة وتنشيط المخرج وترميز الهدف — جدول التوافق في الوحدة 13."),
                   ("`metrics`", "ما يُراقب ويُسجل (لا يؤثر في التدريب)", "`'accuracy'`, `'mae'`, `keras.metrics.AUC()`, `Precision()`, `Recall()`", "المقياس الذي يعكس القرار (وحدة 18). يمكن قائمة بعدة مقاييس.")],
                  ["code", "rtl", "code", "rtl"])
    intuition("الخسارة **تُشتق**، المقياس **يُقرأ**. الدقة غير قابلة للاشتقاق (درجة سُلّم) فلا تصلح خسارة، لكنها ما يهمك — فتضعها في `metrics`. الخسارة تختارها الرياضيات، والمقياس يختاره القرار.")
    h2("النص مقابل الكائن", "String vs object")
    st.markdown("""
- `loss="mse"` و`optimizer="adam"` اختصارات بإعدادات افتراضية (Adam بـ η = 0.001).
- الكائن `keras.optimizers.Adam(learning_rate=3e-4)` أو `keras.losses.BinaryCrossentropy(from_logits=True)` عندما تحتاج **تغيير معلمة**: معدل التعلم، القصّ (`clipnorm=`)، `from_logits`، أوزان الفئات…
- في المقرر نفضّل الكائن في الأمثلة الجدية لأن **معدل التعلم قرار صريح** لا افتراضًا مخفيًا.
""")
    code_lab(CodeLab(
        key="keras_compile", title_ar="compile لا يحسب شيئًا — وإعادة compile لا تفقد الأوزان", code=CODE, level="B",
        before=Before(goal_ar="إثبات أن compile يربط فقط: الأوزان لا تتغير به، وتتغير بـ fit، وتبقى عند إعادة compile بمحسّن آخر.", stage_ar="Keras: compile.",
                      inputs_ar="تصنيف 3 فئات بهدف أعداد صحيحة.", expected_ar="compiled False ثم True؛ الأوزان تتغير بـ fit فقط؛ تبقى بعد إعادة compile.",
                      objects_ar="`keras.optimizers.Adam`, `keras.optimizers.SGD`, نص الخسارة."),
        explain=[("7", "الهدف أعداد صحيحة 0/1/2 (لا one-hot) → الخسارة `sparse_categorical_crossentropy` (وحدة 13)."), ("11-16", "compile: محسّن ككائن (η صريح)، خسارة كنص، مقياس كنص."),
                 ("17-18", "فحص ما رُبط: اسم المحسّن ومعدل تعلمه وكائن الخسارة."), ("20-23", "نسخ الأوزان قبل fit ومقارنتها بعده: compile لم يغيّرها، fit غيّرها."), ("26-27", "إعادة compile بمحسّن آخر — الأوزان محفوظة. مفيد لتغيير η أو المحسّن في منتصف مشروع (تُفقد حالة Adam فقط).")],
        run=run_printed(CODE),
        after_ar="- `history keys` تحوي `loss` و`accuracy` لأننا طلبنا المقياس؛ لو أضفنا `validation_data` لظهر `val_loss` و`val_accuracy`.\n- إعادة compile تصفّر **حالة المحسّن** (متوسطات Adam المتحركة) لا الأوزان.",
    ))
    debugging_note("`ValueError: Shapes (32, 1) and (32, 3) are incompatible` عند fit = عدم توافق الخسارة/الهدف: `categorical_crossentropy` يتوقع one-hot بشكل (batch, 3) بينما هدفك أعداد صحيحة (batch,) أو (batch, 1). الحل: `sparse_categorical_crossentropy`، أو `to_categorical(y)`. (تفصيل في صفحة الأخطاء.)")
    common_mistake("`Dense(1, activation='sigmoid')` مع `loss='categorical_crossentropy'`: لا خطأ برمجي لكن الرياضيات خاطئة (وحدة 13). التوافق: sigmoid ↔ binary_crossentropy؛ softmax ↔ (sparse_)categorical_crossentropy؛ بلا تنشيط ↔ mse/mae/huber.")
    quiz("keras.compile", [
        Q("`compile()`…", ["يغيّر الأوزان", "يربط المحسّن والخسارة والمقاييس فقط", "يحفظ النموذج"], 1, "لا حساب."),
        Q("هدف أعداد صحيحة 0..K−1 مع softmax: الخسارة", ["categorical_crossentropy", "sparse_categorical_crossentropy", "mse"], 1, "sparse."),
        Q("المقياس في `metrics=`…", ["يُشتق ويُقلَّل", "يُراقب فقط", "يحل محل الخسارة"], 1, "لا يؤثر في التدريب."),
        Q("لتغيير معدل التعلم عن الافتراضي…", ["`optimizer='adam'`", "`optimizer=keras.optimizers.Adam(learning_rate=3e-4)`", "لا يمكن"], 1, "الكائن."),
    ])
    takeaway("compile = ربط (محسّن، خسارة، مقاييس). الخسارة تُشتق، المقياس يُقرأ. استخدم الكائن عندما تحتاج معلمة صريحة. التوافق خسارة/تنشيط/هدف من الوحدة 13.")
    lesson_footer(LESSON, ["ثلاثة معاملات وأدوارها.", "نص مقابل كائن.", "إعادة compile تحفظ الأوزان."])
