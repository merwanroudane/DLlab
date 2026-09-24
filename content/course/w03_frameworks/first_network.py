import plotly.graph_objects as go_fig
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, debugging_note, definition, interpretation_note, intuition, takeaway, warning_note, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from content.course.w03_frameworks._viz import WORKFLOW, fit_frames, fit_trace, params_svg, tensor_svg, workflow_svg
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="course.w03.first_network",
    title_ar="أول شبكة عصبية في Keras: كل سطر، كل معامل، كل مخرج",
    title_en="Your First Neural Network in Keras: Every Line, Every Argument, Every Output",
    module="course.w03",
    order=2,
    prerequisites=["course.w03.overview", "foundations.frameworks.keras.code_lab_standard", "foundations.frameworks.keras.output_explorer"],
    objectives_ar=[
        "مراجعة الموتر (shape/rank/dtype/device) بصريًا قبل استخدام أي API.",
        "رؤية سير عمل Keras الستة (Sequential → summary → compile → fit → evaluate → predict) وما يُنشئه كل منها.",
        "عدّ المعلمات يدويًا (209) ومطابقته مع summary.",
        "مشاهدة fit من الداخل دفعةً دفعة (تتبّع حقيقي) وقراءة History.",
        "بناء الشبكة على تعثر القروض وتحويل الاحتمالات إلى قرار وتشخيص المخرجات.",
    ],
    terms=["tensor", "shape", "rank", "dtype", "epoch", "batch", "batch_size", "iteration", "loss", "keras", "fit", "callback", "early_stopping"],
    labs=["labs.tf_tensor_explorer", "labs.threshold_lab", "labs.parameter_counter", "labs.epoch_batch_simulator"],
    difficulty="intermediate",
    summary_ar="tensor → layers → model → summary → compile → fit(x, y, batch_size, epochs, validation_data, shuffle, callbacks, verbose) → History → evaluate → predict → decision. كل واحدة مفكّكة ومحرّكة، وزر «المكافئ في PyTorch».",
)

TENSOR = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, tensorflow as tf
from labs.datasets import loan_default
df = loan_default(n=400, seed=7)
X_np = df[["income", "age", "num_late_payments", "debt_ratio"]].to_numpy("float32")   # float32 دائمًا
t = tf.convert_to_tensor(X_np)
print("shape:", t.shape, "| rank:", int(tf.rank(t)), "| dtype:", t.dtype.name, "| device:", t.device.split("/")[-1])
print("one observation:", t[0].numpy().round(2), "-> shape", t[0].shape, "(rank 1: a vector of 4 features)")
print("a batch of 32  :", t[:32].shape, "(rank 2: batch × features) <- what Dense expects")
print("NaN present?   :", bool(np.isnan(X_np).any()), "-> must be imputed before training")'''

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, keras
from keras import layers, callbacks
from labs.datasets import loan_default
keras.utils.set_random_seed(0)

# --- البيانات (الأسبوع 02: تقسيم ثم تحجيم بإحصاءات التدريب) ---
df = loan_default(n=400, seed=7)
num = df[["income", "age", "num_late_payments", "debt_ratio"]].to_numpy("float32")
cat = pd.get_dummies(df[["city", "employment"]]).to_numpy("float32")
y = df["defaulted"].to_numpy("float32")
idx = np.random.default_rng(0).permutation(len(df)); tr, va, te = idx[:240], idx[240:320], idx[320:]
med = np.nanmedian(num[tr], 0); num = np.where(np.isnan(num), med, num)
mu, sd = num[tr].mean(0), num[tr].std(0) + 1e-8
X = np.concatenate([(num - mu) / sd, cat], 1)                        # (400, 11)

# --- 1) الطبقات والنموذج ---
model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),                # ملاحظة واحدة = 11 خاصية
    layers.Dense(16, activation="relu"),              # (None, 16): 11*16+16 = 192
    layers.Dense(1, activation="sigmoid"),            # (None, 1) : 16+1 = 17 -> احتمال التعثر
], name="loan_mlp")
model.summary()

# --- 2) compile: المحسّن (كيف نحدّث)، الخسارة (ما نقلّل)، المقاييس (ما نراقب) ---
model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-3), loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])

# --- 3) fit: الحلقة كاملة بمعاملاتها ---
es = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
history = model.fit(
    X[tr], y[tr],                       # x, y
    batch_size=32,                      # 240/32 -> 8 تحديثات لكل حقبة
    epochs=100,                         # حد أعلى؛ الإيقاف المبكر يقرر
    validation_data=(X[va], y[va]),     # تحقق منفصل (لا validation_split هنا: لدينا تقسيمنا)
    shuffle=True,                       # خلط كل حقبة
    callbacks=[es],                     # الإيقاف المبكر
    verbose=0,                          # صامت هنا؛ استخدم 2 في الدفتر
)
# --- 4) History ---
h = history.history
best = int(np.argmin(h["val_loss"])) + 1
print(f"\\nepochs run: {len(h['loss'])} | best epoch: {best} | keys: {list(h)}")
print("val_loss by epoch:", np.round(h["val_loss"][:6], 3), "...")

# --- 5) evaluate: على الاختبار مرة واحدة ---
loss, acc, auc = model.evaluate(X[te], y[te], verbose=0)
print(f"test: loss={loss:.4f} accuracy={acc:.3f} auc={auc:.3f} | baseline accuracy={max(y[te].mean(), 1 - y[te].mean()):.3f}")

# --- 6) predict: احتمالات -> قرار ---
p = model.predict(X[te][:5], verbose=0)
print("predict shape:", p.shape, "| probabilities:", p.ravel().round(3), "| decision@0.5:", (p.ravel() >= 0.5).astype(int), "| truth:", y[te][:5].astype(int))'''

FUNCTIONAL = '''inp = keras.Input(shape=(11,))
h = layers.Dense(16, activation="relu")(inp)
out = layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(inp, out)          # نفس الشبكة بالضبط — الرسم بدل القائمة'''


def render() -> None:
    lesson_header(LESSON)
    why("الوحدة 21 شرحت كل قطعة في صفحتها. هنا القطع كلها في **ملف واحد** كما ستكتبه في مشروعك — مع الأسئلة الأحد عشر قبله، وتحريك لكل مرحلة، وتفكيك كل مخرج بعده، وزر ينقلك إلى المكافئ في PyTorch.")

    # ------------------------------------------------------------------ tensors
    h2("قبل أي API: الموتر بصريًا", "Before any API: tensors, visually")
    tcaps = [
        ("**عدد واحد (رتبة 0)**: نسبة دين العميل الأول. شكله `()` — لا محاور.", "scalar"),
        ("**متجه (رتبة 1)**: عميل واحد = 4 خصائص. شكله `(4,)`. هذا **ملاحظة** لا دفعة؛ Dense لا يقبله مباشرة.", "vector"),
        ("**مصفوفة (رتبة 2)**: دفعة من العملاء. المحور 0 = الدفعة (الصفوف)، المحور 1 = الخصائص. شكله `(32, 4)` — هذا ما تتوقعه `Dense`.", "matrix = batch"),
        ("**بعد الترميز الأحادي** `one-hot` للمدينة والحالة الوظيفية: 4 عددية + 7 أعمدة 0/1 = **11 خاصية**. لهذا `Input(shape=(11,))`.", "one-hot → 11"),
        ("**النوع والجهاز**: `float32` (نصف ذاكرة float64 وهو افتراضي Keras) على `CPU:0` (أو GPU في Colab). رتبة + شكل + نوع + جهاز = الخصائص الأربع لكل موتر.", "dtype · device"),
    ]
    animation_player("w03_tensor", [Frame(tensor_svg(i), caption(c), action=a, highlight=min(i, 3)) for i, (c, a) in enumerate(tcaps)],
                     title_ar="من رقم إلى دفعة: الرتب الثلاث", stages=["rank 0", "rank 1", "rank 2", "(B, 11)"], interval_ms=2400)
    code_lab(CodeLab(
        key="w03_tensor", title_ar="بيانات القروض كموتر: shape / rank / dtype / device", code=TENSOR, level="B",
        before=Before(goal_ar="تحويل الجدول إلى موتر وقراءة خصائصه الأربع، وتحديد شكل الملاحظة الواحدة وشكل الدفعة الذي تتوقعه Dense.", stage_ar="الأسبوع 03: الموتر.", inputs_ar="4 أعمدة عددية من loan_default.", expected_ar="shape (400, 4)، رتبة 2، float32، CPU؛ الملاحظة رتبة 1؛ الدفعة (32, 4)؛ NaN موجودة."),
        explain=[("5-6", "`float32` صراحةً عند التحويل: الافتراضي في Keras، ونصف ذاكرة float64."), ("7", "الخصائص الأربع في سطر: هذا ما تفحصه قبل أي نموذج."), ("8-9", "ملاحظة واحدة متجه (رتبة 1)؛ Dense يريد دفعة (رتبة 2) — لذلك `x[None, :]` للملاحظة الواحدة."), ("10", "NaN في `debt_ratio` (متعمدة في البيانات): يجب ملؤها قبل التدريب وإلا خسارة nan.")],
        run=run_printed(TENSOR), after_ar="- الرتبة 2 = (batch, features) هي عقد Dense.\n- فحص NaN قبل fit يوفر ساعة تشخيص لاحقًا.",
    ))
    st.button("افتح مستكشف موترات TensorFlow", icon=":material/science:", on_click=go, args=("labs.tf_tensor_explorer",), key="w03_lab_tensor")

    # ------------------------------------------------------------------ the eleven questions
    h2("قبل الكود: الأسئلة الأحد عشر", "Before the code: the eleven questions")
    compare_table(["السؤال", "الجواب"],
                  [("لماذا Keras هنا؟", "مسألة جدولية قياسية؛ نريد fit مع إيقاف مبكر ومقاييس جاهزة."), ("المدخل؟", "11 خاصية: 4 عددية موحَّدة + 7 one-hot؛ دفعة (batch, 11)."), ("الهدف؟", "defaulted ∈ {0,1}؛ نسبة الإيجابيات = 50% (فئتان متوازنتان)."), ("البنية؟", "11 → 16 (ReLU) → 1 (sigmoid)."), ("الأشكال؟", "(None, 11) → (None, 16) → (None, 1)."),
                   ("التنشيط؟", "ReLU مخفي؛ sigmoid للمخرج الثنائي."), ("الخسارة؟", "binary_crossentropy (ثنائي + sigmoid)."), ("المقاييس؟", "accuracy للمقارنة بخط الأساس، AUC لجودة الترتيب."), ("المحسّن؟", "Adam بـ η = 3e-3."), ("حجم الدفعة؟", "32: 8 تحديثات لكل حقبة (240 ÷ 32 = 7.5 ← 8)."), ("عدد الحقب؟", "100 حد أعلى؛ EarlyStopping(patience=8) يقرر.")],
                  ["rtl", "rtl"])

    # ------------------------------------------------------------------ workflow
    h2("سير عمل Keras في ست خطوات", "The Keras workflow in six calls")
    wcaps = [
        "**`Sequential`**: ينشئ كائن النموذج وطبقاته. الأوزان موجودة الآن لكنها **عشوائية** (تهيئة Glorot افتراضيًا): النموذج جاهز للتنبؤ لكنه لا يعرف شيئًا.",
        "**`summary()`**: لا يحسب شيئًا من البيانات؛ يطبع أشكال المخرجات وعدد المعلمات لكل طبقة. أداة فحص قبل التدريب.",
        "**`compile()`**: يربط ثلاثة كائنات بالنموذج: **المحسّن** (كيف نحدّث)، **الخسارة** (ما نصغّر)، **المقاييس** (ما نراقب فقط، لا تُصغَّر).",
        "**`fit()`**: الحلقة كاملة — حقب × دفعات × (أمامي، خسارة، تدرج، تحديث) + تحقق كل حقبة. يعدّل النموذج والمحسّن في مكانهما ويعيد `History`.",
        "**`evaluate()`**: تمرير أمامي فقط على بيانات محجوزة، ويعيد `[loss, metric1, metric2…]` بترتيب compile. مرة واحدة على الاختبار.",
        "**`predict()`**: احتمالات لصفوف جديدة. لا خسارة (لا يحتاج y) ولا تحديث. القرار = احتمال + عتبة تختارها أنت.",
    ]
    animation_player("w03_workflow", [Frame(workflow_svg(i), caption(c), action=WORKFLOW[i][0], highlight=i) for i, c in enumerate(wcaps)],
                     title_ar="ما يُنشئه كل استدعاء", stages=[w[0] for w in WORKFLOW], interval_ms=2600)

    # ------------------------------------------------------------------ parameter count
    h3("من أين جاء الرقم 209؟", "Where does 209 come from?")
    pcaps = ["**طبقة المدخل** تثبّت الشكل فقط `(None, 11)`؛ `None` = حجم دفعة حر. لا معلمات.",
             "**Dense(16)**: كل وحدة من 16 تتصل بكل خاصية من 11 ⇒ مصفوفة أوزان 11×16 = 176، زائد انحياز لكل وحدة (16) = **192**.",
             "**Dense(1)**: وحدة المخرج تتصل بالوحدات الـ 16 ⇒ 16 وزنًا + انحياز واحد = **17**.",
             "**المجموع 209** — نفس الرقم الذي يطبعه `summary()` حرفيًا. إن اختلف رقمك فشكل المدخل ليس ما تظن."]
    animation_player("w03_params", [Frame(params_svg(i), caption(c), action=["Input", "Dense(16)", "Dense(1)", "summary"][i]) for i, c in enumerate(pcaps)],
                     title_ar="عدّ المعلمات خلية خلية", interval_ms=2400)
    equation(r"\text{params}(\text{Dense}) = n_{\text{in}} \times n_{\text{out}} + n_{\text{out}}",
             [(r"n_{\text{in}}", "عدد الخصائص الداخلة (أو وحدات الطبقة السابقة)."), (r"n_{\text{out}}", "عدد وحدات الطبقة = عدد الانحيازات.")],
             meaning_ar="أوزان كاملة الاتصال + انحياز لكل وحدة. هذه الصيغة وحدها تكفي لفحص أي summary لشبكة كثيفة.",
             example_ar="11×16+16 = 192؛ 16×1+1 = 17؛ المجموع 209 معلمة × 4 بايت (float32) = 836 بايت.",
             dl_link_ar="السطر `Total params: 209 (836.00 B)` في summary = هذا الحساب بالضبط.", title_ar="عدد معلمات طبقة كثيفة")

    # ------------------------------------------------------------------ the file
    h2("الملف الكامل: من الجدول إلى القرار", "The complete file")
    code_lab(CodeLab(
        key="w03_first", title_ar="أول شبكة: من الجدول إلى القرار في ملف واحد", code=CODE, level="C",
        before=Before(goal_ar="الشبكة الأولى كاملة على تعثر القروض بكل مراحل Keras، مع تفكيك كل مخرج في «بعد الكود».", stage_ar="الأسبوع 03: Keras end-to-end.",
                      inputs_ar="loan_default(400): 11 خاصية بعد الترميز والتحجيم.", expected_ar="summary بـ 209 معلمة؛ توقف قبل 100 حقبة؛ test accuracy قريبة من خط الأساس أو أعلى قليلًا؛ AUC ≈ 0.65–0.75؛ 5 احتمالات وقراراتها.", prerequisites_ar="الأسبوع 02 (التقسيم/خط الأساس)؛ الوحدة 21 (كل صفحات Keras)."),
        explain=[("7-15", "إعداد البيانات كما في الأسبوع 02: تقسيم بالفهارس، ملء المفقود بوسيط التدريب، توحيد بإحصاءات التدريب، دمج one-hot."), ("18-23", "**Layers → Model**: Input بشكل الملاحظة، طبقة مخفية، مخرج sigmoid. `summary()` يطبع الجدول (تفكيكه في الوحدة 21)."),
                 ("26", "**compile**: Adam بمعدل صريح؛ الخسارة الثنائية؛ مقياسان."), ("29-38", "**fit** بكل معامل وتعليقه: x/y، batch_size، epochs، validation_data، shuffle، callbacks، verbose. الحلقة كاملة تحدث هنا (محرك fit في الوحدة 21)."),
                 ("40-43", "**History**: قاموس قوائم بطول الحقب الفعلية؛ أفضل حقبة من val_loss."), ("46-47", "**evaluate** على الاختبار مرة واحدة، بترتيب [loss, accuracy, auc]، مع خط الأساس بجانبه."), ("50-51", "**predict** يعطي (5, 1) احتمالات؛ العتبة 0.5 تعطي القرار — والعتبة قرار اقتصادي (معمل العتبة).")],
        run=run_printed(CODE),
        after_ar="""- **summary**: 192 + 17 = 209 معلمة؛ `(None, 11)` بُعد الدفعة حر.
- **epochs run < 100**: الإيقاف المبكر عمل؛ الأوزان المستعادة من أفضل حقبة.
- **test vs baseline**: الفرق هو القيمة. AUC ≈ 0.66 يعني ترتيبًا أفضل من العشوائي لكن ليس ممتازًا — بيانات صغيرة (240 للتدريب) وشبكة أكبر من الحاجة؛ قارن باللوجستي في الأسبوع 02 (AUC ≈ 0.73 بخصائص أقل).
- **predict**: احتمالات لا فئات؛ قراءة الصف: p = 0.7 وقرار 1 وحقيقة 0 = إنذار كاذب.
- **التشخيص**: لو كانت accuracy = baseline بالضبط → تنبؤ بفئة واحدة (تحقق من التحجيم وη). لو val_loss يصعد من الحقبة 3 → قلّل الوحدات أو أضف Dropout.""",
    ))

    # ------------------------------------------------------------------ fit from the inside
    h2("داخل fit: دفعة بعد دفعة (تتبّع حقيقي)", "Inside fit(): batch by batch (a real trace)")
    st.markdown("سجّلنا ما يحدث داخل `model.fit` بـ `Callback` على شبكة صغيرة (هلالان، 120 ملاحظة تدريب، دفعات 32، SGD بـ η = 0.5): كل مربع = دفعة = **تحديث واحد** لكل الأوزان.")
    tr = fit_trace()
    if tr and "events" in tr:
        ff = fit_frames(tr)
        animation_player("w03_fit", [Frame(svg, caption(c), action=a, values=v) for svg, c, a, v in ff],
                         title_ar=f"fit(epochs=5, batch_size=32): {tr['steps_per_epoch']} دفعات × 5 حقب", interval_ms=900)
        equation(r"\text{steps per epoch} = \left\lceil \frac{n_{\text{train}}}{\text{batch\_size}} \right\rceil, \qquad \text{total updates} = \text{epochs} \times \text{steps}",
                 [(r"n_{\text{train}}", "عدد ملاحظات التدريب."), (r"\lceil\cdot\rceil", "تقريب للأعلى: الدفعة الأخيرة قد تكون أصغر.")],
                 meaning_ar="الحقبة = مرور كامل؛ الخطوة = دفعة واحدة = تحديث واحد. «4/4» في شريط Keras تعني الخطوة الرابعة من أربع.",
                 example_ar=f"{tr['n']} ÷ 32 = {tr['n'] / 32:.2f} ← {tr['steps_per_epoch']} خطوات؛ 5 حقب ⇒ {5 * tr['steps_per_epoch']} تحديثًا. في ملف القروض: 240 ÷ 32 ← 8 خطوات.",
                 dl_link_ar="الدفعة الأخيرة هنا 120 − 3×32 = 24 ملاحظة فقط.", title_ar="الخطوات والحقب")
        h = tr["history"]
        fh = go_fig.Figure()
        ep = list(range(1, len(h["loss"]) + 1))
        fh.add_scatter(x=ep, y=h["loss"], mode="lines+markers", name="loss", line=dict(color="#7C3AED", width=3))
        fh.add_scatter(x=ep, y=h["val_loss"], mode="lines+markers", name="val_loss", line=dict(color="#D97706", width=3))
        fh.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="epoch", yaxis_title="loss", legend=dict(orientation="h", x=0, y=1.18))
        st.plotly_chart(fh, width="stretch", key="w03_hist_fig")
        interpretation_note(f"`history.history` قاموس مفاتيحه {list(h)} وكل قائمة بطول عدد الحقب ({len(h['loss'])}). "
                            "loss = متوسط خسائر الدفعات **أثناء** الحقبة (والأوزان تتغير)، وval_loss = خسارة التحقق **بعد** الحقبة. لذلك قد تكون val_loss أقل من loss في الحقب الأولى — ليس خطأ.")
        with st.expander("السجل الخام كما يطبعه Keras (`verbose=2`)", icon=":material/terminal:"):
            st.code(tr["log"].strip() or "(no log)", language="text")
    else:
        warning_note(f"TensorFlow غير متاح في بيئة التشغيل هذه ({(tr or {}).get('error', '')}). شغّل التطبيق على Python 3.11–3.13 لرؤية التتبّع الحقيقي.")

    h3("تمهيد Functional API", "Functional API preview")
    st.code(FUNCTIONAL, language="python")
    st.markdown("نفس الشبكة كرسم من الاستدعاءات — تحتاجه عند التفرّع أو تعدد المدخلات (الوحدة 21: الطبقات والنماذج).")
    compare_table(["", "Sequential", "Functional"],
                  [("الشكل", "قائمة طبقات بالترتيب", "رسم: `h = Layer(...)(x)`"), ("مدخلات متعددة/تفرّع", "لا", "نعم"), ("متى؟", "شبكة خطية بسيطة (معظم المقرر)", "مدخل جدولي + صورة، أو اتصالات تخطٍّ")],
                  ["rtl", "code", "code"])
    with st.container(horizontal=True):
        st.button("المكافئ في PyTorch", icon=":material/swap_horiz:", type="primary", on_click=go, args=("course.w03.pytorch_equivalent",), key="w03_go_pt")
        st.button("تفكيك summary", icon=":material/table_view:", on_click=go, args=("foundations.frameworks.keras.summary_deconstruction",), key="w03_go_summary")
        st.button("مستكشف المخرجات", icon=":material/visibility:", on_click=go, args=("foundations.frameworks.keras.output_explorer",), key="w03_go_outputs")
        st.button("معمل العتبة", icon=":material/science:", on_click=go, args=("labs.threshold_lab",), key="w03_lab_thr")
        st.button("معمل عدّ المعلمات", icon=":material/science:", on_click=go, args=("labs.parameter_counter",), key="w03_lab_params")
    intuition("كل سطر في هذا الملف يقابل خطوة في الأسس: البيانات (8)، الطبقات (10–12)، compile (13، 15)، fit (16–17)، History (19)، evaluate (18)، predict (12، 18). لا يوجد سطر بلا وحدة تفسره.")
    debugging_note("**الأخطاء الخمسة الشائعة** في هذا الملف بالذات: (1) شكل المدخل ≠ X.shape[1]؛ (2) ترميز الهدف (y أعداد صحيحة مع categorical_crossentropy)؛ (3) sigmoid مع خسارة متعددة أو العكس؛ (4) float64/object من pandas؛ (5) Colab بلا GPU مفعّل (يعمل على CPU أبطأ). كلها مشروحة برسائلها الحقيقية في الوحدة 21 والمعرض.")
    common_mistake("`validation_split=0.2` هنا بدل `validation_data`: البيانات مخلوطة فلا ضرر، لكن في مشروعك قد تكون مرتبة بالتاريخ أو الفئة — `validation_split` يأخذ **آخر** 20% من الصفوف قبل الخلط. احتفظ بتقسيمك الصريح كما في الأسبوع 02.")
    quiz("w03.first", [
        Q("عدد معلمات Dense(16) على 11 خاصية:", ["176", "192", "27"], 1, "11×16+16."),
        Q("`epochs=100` مع EarlyStopping(patience=8):", ["100 حقبة دائمًا", "حد أعلى؛ يتوقف بعد 8 حقب بلا تحسن", "8 حقب"], 1, ""),
        Q("`model.evaluate` هنا يعيد…", ["[loss]", "[loss, accuracy, auc]", "الاحتمالات"], 1, "بترتيب compile."),
        Q("p = 0.7 والقرار 1 والحقيقة 0:", ["TP", "FP (إنذار كاذب)", "FN"], 1, ""),
        Q("1000 ملاحظة تدريب، batch_size = 64: عدد الخطوات في الحقبة", ["15", "16", "64"], 1, "⌈1000/64⌉ = 16."),
        Q("موتر بشكل (32, 11) رتبته", ["1", "2", "11"], 1, "محوران."),
        Q("المقاييس في compile…", ["تُصغَّر مع الخسارة", "تُراقب فقط", "تغيّر المحسّن"], 1, "الخسارة وحدها تُصغَّر."),
        Q("بعد `Sequential([...])` وقبل fit، الأوزان…", ["أصفار", "عشوائية (مهيأة)", "غير موجودة"], 1, "التهيئة تحدث عند البناء."),
    ])
    takeaway("ملف واحد من الموتر إلى القرار. ست استدعاءات لكل منها دور. 209 = Σ(in×out + out). fit = حقب × دفعات × تحديث. 11 سؤالًا قبل الكود، وتفكيك كل مخرج بعده.")
    lesson_footer(LESSON, ["الموتر بصريًا (تحريك) + مراجعة بالكود.", "سير العمل الستة (تحريك).", "عدّ المعلمات (تحريك).", "الملف الكامل.", "fit دفعةً دفعة بتتبّع حقيقي + History.", "Functional كتمهيد والروابط."])
