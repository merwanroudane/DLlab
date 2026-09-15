import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, intuition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.frameworks.keras.evaluate_predict",
    title_ar="evaluate() وpredict(): القياس والتنبؤ",
    title_en="evaluate() & predict()",
    module="foundations.frameworks",
    parent="foundations.frameworks.keras",
    order=7,
    prerequisites=["foundations.frameworks.keras.fit", "foundations.eval.splits_metrics", "foundations.forward.logits_probability_threshold"],
    objectives_ar=["الفرق بين evaluate (خسارة + مقاييس على بيانات موسومة) وpredict (مخرجات خام على بيانات بلا وسم).", "قراءة قائمة evaluate بترتيبها، وشكل مصفوفة predict وتحويلها إلى قرارات.", "معرفة أن كليهما تمرير أمامي بلا تدرج وبوضع الاستدلال."],
    terms=["probability", "softmax"],
    labs=["labs.threshold_lab"],
    difficulty="beginner",
    summary_ar="evaluate(X, y) → [loss, metric1, …] بترتيب compile. predict(X) → مصفوفة مخرجات الطبقة الأخيرة بشكل (n, units)؛ الاحتمالات تحتاج عتبة أو argmax لتصبح فئات.",
)

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, keras
from keras import layers
keras.utils.set_random_seed(0)
rng = np.random.default_rng(0)
X = rng.normal(size=(600, 4)).astype("float32")
y = (np.argmax(X[:, :3] + 0.5 * rng.normal(size=(600, 3)), axis=1)).astype("int32")    # 3 فئات
X_tr, y_tr, X_te, y_te = X[:500], y[:500], X[500:], y[500:]

model = keras.Sequential([layers.Input(shape=(4,)), layers.Dense(16, activation="relu"), layers.Dense(3, activation="softmax")])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_tr, y_tr, epochs=15, batch_size=32, verbose=0)

# evaluate: يحتاج y؛ يعيد [loss, *metrics] بترتيب compile
res = model.evaluate(X_te, y_te, verbose=0)
print("evaluate ->", np.round(res, 4), "| names:", model.metrics_names)
print("as dict  ->", {k: round(float(v), 4) for k, v in zip(model.metrics_names, res)})

# predict: لا يحتاج y؛ يعيد مخرج الطبقة الأخيرة
P = model.predict(X_te, verbose=0)
print("predict shape:", P.shape, "| dtype:", P.dtype, "| first row:", P[0].round(3), "sum =", P[0].sum().round(3))
y_pred = P.argmax(axis=1)                                     # softmax -> فئة
print("classes      :", y_pred[:10], "| true:", y_te[:10])
print("manual acc   :", (y_pred == y_te).mean().round(4), "== evaluate acc:", round(res[1], 4))
print("confidence   : max prob mean =", P.max(1).mean().round(3), "| lowest-confidence sample:", int(P.max(1).argmin()), P[P.max(1).argmin()].round(3))
one = model.predict(X_te[:1], verbose=0)                       # ملاحظة واحدة: يجب أن تبقى ثنائية البعد (1, 4)
print("single sample -> shape", one.shape)'''


def render() -> None:
    lesson_header(LESSON)
    definition("**`evaluate(X, y)`**: تمرير أمامي على بيانات **موسومة** بوضع الاستدلال، يعيد الخسارة والمقاييس كما عُرّفت في compile. **`predict(X)`**: تمرير أمامي على بيانات **بلا وسم**، يعيد مخرج الطبقة الأخيرة (احتمالات إن كان التنشيط sigmoid/softmax، قيمًا خطية للانحدار). كلاهما بلا تدرج وبلا تحديث، وDropout/BN في وضع الاستدلال تلقائيًا.")
    compare_table(["", "evaluate", "predict"],
                  [("المدخل", "X و y", "X فقط"), ("المخرج", "قائمة `[loss, metric₁, …]` (أو عدد واحد بلا مقاييس)", "مصفوفة `(n, units_of_last_layer)`"), ("الغرض", "تقرير الأداء على تحقق/اختبار", "استخدام النموذج: قرارات، تحليل أخطاء، إنتاج"),
                   ("الدفعات", "نعم (`batch_size=32` افتراضيًا)؛ النتيجة على كل البيانات", "نعم؛ يجمّع المخرجات"), ("وضع الطبقات", "استدلال (Dropout معطّل، BN بالمتحركة)", "استدلال"), ("التدرج", "لا", "لا")],
                  ["rtl", "rtl", "rtl"])
    code_lab(CodeLab(
        key="keras_eval_pred", title_ar="evaluate وpredict على مجموعة اختبار، ثم إعادة حساب الدقة يدويًا", code=CODE, level="B",
        before=Before(goal_ar="قياس نموذج مدرّب على اختبار محجوز، قراءة قائمة evaluate بأسمائها، تحويل مخرج predict إلى فئات، والتحقق من أن الدقة اليدوية تساوي دقة evaluate.", stage_ar="Keras: evaluate / predict.",
                      inputs_ar="100 ملاحظة اختبار × 4 خصائص، 3 فئات.", expected_ar="قائمة من عنصرين [loss, accuracy]، مصفوفة (100, 3) بصفوف مجموعها 1، ودقة يدوية مطابقة."),
        explain=[("14-16", "`evaluate` يعيد قائمة بترتيب: الخسارة أولًا ثم المقاييس بترتيب compile. `metrics_names` يعطي الأسماء — اربطهما بـ zip."), ("19-20", "`predict` يعيد (100, 3) float32: احتمال لكل فئة، الصف يجمع إلى 1 (softmax، وحدة 12)."),
                 ("21-23", "argmax على المحور 1 → الفئة. الدقة اليدوية = evaluate: لا سحر في evaluate."), ("24", "تحليل الثقة: أدنى احتمال أقصى = الملاحظة الأكثر غموضًا. مادة لتحليل الأخطاء."), ("25-26", "ملاحظة واحدة يجب أن تبقى (1, 4) لا (4,): بُعد الدفعة إلزامي.")],
        run=run_printed(CODE),
        after_ar="- `evaluate` = خسارة ومقاييس **على كل مجموعة الاختبار** (لا متوسط تراكمي كما في fit).\n- `predict` يعطي احتمالات لا فئات؛ التحويل قرارك: argmax للمتعدد، عتبة للثنائي (وحدة 18: العتبة).\n- الدقة اليدوية = دقة evaluate: تحقق يمكنك تكراره دائمًا.",
    ))
    intuition("`predict` لا «يقرر». يعطيك ما تعطيه الطبقة الأخيرة. للثنائي: `(P >= 0.5)` عتبة افتراضية قد لا تناسب كلفتك (وحدة 18). للمتعدد: argmax. للانحدار: القيمة نفسها بوحدات الهدف (بعد عكس التحجيم إن حجّمت y).")
    debugging_note("`predict(x)` يرفع `ValueError: ... expected shape (None, 4), found (4,)`: مررت ملاحظة واحدة كمتجه. الحل: `x[None, :]` أو `x.reshape(1, -1)`. وللمخرج: `predict` يعيد (1, units) — خذ `[0]`.")
    common_mistake("استدعاء `model(X)` بدل `predict(X)` على مجموعة ضخمة: `model(X)` يمرر كل شيء دفعة واحدة (ذاكرة) ويعيد موترًا لا NumPy. `predict` يقسّم دفعات ويعيد NumPy. للدفعات الصغيرة داخل حلقات مخصصة `model(x, training=False)` مناسب.")
    quiz("keras.evalpred", [
        Q("`evaluate` يعيد…", ["الاحتمالات", "[loss, metrics…] بترتيب compile", "الفئات"], 1, "خسارة ومقاييس."),
        Q("مخرج `predict` لـ Dense(3, softmax) على 100 ملاحظة", ["(100,)", "(100, 3)", "(3,)"], 1, "n × units."),
        Q("Dropout أثناء evaluate/predict…", ["فعّال", "معطّل (وضع الاستدلال)", "عشوائي"], 1, "Keras يتولى ذلك."),
        Q("ملاحظة واحدة بـ 4 خصائص تُمرَّر بشكل", ["(4,)", "(1, 4)", "(4, 1)"], 1, "بُعد الدفعة."),
    ])
    takeaway("evaluate = خسارة + مقاييس على بيانات موسومة؛ predict = مخرج الطبقة الأخيرة. كلاهما استدلال بلا تدرج. الاحتمال → قرار بعتبة أو argmax، وهذا قرارك.")
    lesson_footer(LESSON, ["الفرق والجدول.", "قراءة القائمة والمصفوفة.", "بُعد الدفعة إلزامي."])
