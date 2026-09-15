import streamlit as st

from components.callouts import common_mistake, intuition, practical_note, takeaway, why
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q
from components.week import post_test
from core.models import Lesson
from core.routing import go as goto

LESSON = Lesson(
    id="course.w14.notebook_template",
    title_ar="قالب دفتر المشروع (جدولي/صور/تسلسل) والاختبار البعدي",
    title_en="The Project Notebook Template (Tabular / Images / Sequences) — & Post-test",
    module="course.w14",
    order=3,
    prerequisites=["course.w14.project_guide", "course.w03.first_network"],
    objectives_ar=["دفتر جاهز بخلايا مرقّمة تطابق مراحل المشروع، بثلاث نسخ للمدخل (جدول/صور/تسلسل).", "نقاط الاستبدال المطلوبة ومكان كل قرار.", "الاختبار البعدي."],
    terms=["reproducibility"],
    difficulty="intermediate",
    summary_ar="خلية لكل مرحلة: إعداد وبذرة ونسخ → تحميل ووصف → تشخيص → معالجة → تقسيم → خط أساس → نموذج (بحسب النوع) → تدريب بحفظ → منحنيات → تقييم واحد → تفسير → ملخص الاستنساخ. انسخه إلى Colab.",
)

TEMPLATE = '''# ============================================================
# مشروع التعلم العميق — <عنوان المشروع>   |   <الاسم>   |   <التاريخ>
# ============================================================
# [0] الإعداد والاستنساخ
import os, sys, json, time, numpy as np, pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras, tensorflow as tf
from keras import layers, callbacks
SEED = 0; keras.utils.set_random_seed(SEED); np.random.seed(SEED)
print({"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "tensorflow": tf.__version__, "keras": keras.__version__,
       "GPU": tf.config.list_physical_devices("GPU")})
DATA_PATH = "data/my_dataset.csv"          # <-- استبدل (Drive في Colab: /content/drive/MyDrive/...)
OUT_DIR = "outputs"; os.makedirs(OUT_DIR, exist_ok=True)

# [1] تعريف المسألة (نص في التقرير + ثوابت هنا)
TASK = "classification"                    # "regression" | "classification" | "sequence"
TARGET = "defaulted"                       # اسم عمود الهدف
UNIT = "0/1"                               # وحدة الهدف (للتقرير)
METRIC = "recall@precision>=0.6"           # المقياس المرتبط بالقرار

# [2] تحميل ووصف البيانات
df = pd.read_csv(DATA_PATH)
print(df.shape); print(df.dtypes); print(df.describe(include="all").T.head(20))

# [3] تشخيص البيانات
print("missing per column:\\n", df.isna().mean().round(3))
print("duplicates:", df.duplicated().sum())
print("target balance:\\n", df[TARGET].value_counts(normalize=True) if TASK == "classification" else df[TARGET].describe())
# فحص التسريب: أي عمود يرتبط بالهدف بشكل مريب؟ (ارتباط > 0.9 أو تعريف يحوي الهدف)
num_cols = [c for c in df.select_dtypes("number").columns if c != TARGET]
print("suspicious correlations:", df[num_cols + [TARGET]].corr()[TARGET].abs().sort_values(ascending=False).head(5).round(2).to_dict())

# [4]+[5] التقسيم أولًا ثم المعالجة بإحصاءات التدريب
cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != TARGET]
y = df[TARGET].to_numpy("float32")
idx = np.random.default_rng(SEED).permutation(len(df))                  # للسلاسل الزمنية: لا خلط — فهارس زمنية
n_tr, n_va = int(0.6 * len(df)), int(0.2 * len(df))
tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
num = df[num_cols].to_numpy("float32")
med = np.nanmedian(num[tr], 0); num = np.where(np.isnan(num), med, num)
mu, sd = num[tr].mean(0), num[tr].std(0) + 1e-8
cat = pd.get_dummies(df[cat_cols]).to_numpy("float32") if cat_cols else np.zeros((len(df), 0), "float32")
X = np.concatenate([(num - mu) / sd, cat], 1); feature_names = num_cols + list(pd.get_dummies(df[cat_cols]).columns if cat_cols else [])
print("X:", X.shape, "| splits:", len(tr), len(va), len(te))
# --- للصور: X = images.astype("float32") / 255 ; X = X[..., None] إن كانت رمادية ; y أعداد صحيحة
# --- للتسلسل: z = (s - s[:n_tr].mean()) / s[:n_tr].std() ; X, y = make_windows(z, W) ; تقسيم زمني بالفهرس

# [6] خطوط الأساس (على التحقق)
from sklearn.linear_model import LogisticRegression, LinearRegression
if TASK == "classification":
    base = max(y[va].mean(), 1 - y[va].mean()); lin = LogisticRegression(max_iter=500).fit(X[tr], y[tr]); lin_score = lin.score(X[va], y[va])
    print(f"baseline majority acc={base:.3f} | logistic acc={lin_score:.3f}")
elif TASK == "regression":
    base = np.sqrt(np.mean((y[va] - y[tr].mean()) ** 2)); lin = LinearRegression().fit(X[tr], y[tr]); lin_score = np.sqrt(np.mean((lin.predict(X[va]) - y[va]) ** 2))
    print(f"baseline mean RMSE={base:.3f} | linear RMSE={lin_score:.3f} ({UNIT})")
# --- للتسلسل: naive = X_va[:, -1, 0] ; RMSE(naive, y_va)

# [7] البنية — أجب عن الأسئلة الأحد عشر في التقرير قبل هذه الخلية
def build():
    if TASK == "classification":
        m = keras.Sequential([layers.Input(shape=(X.shape[1],)), layers.Dense(32, activation="relu"), layers.Dropout(0.2), layers.Dense(1, activation="sigmoid")])
        m.compile(optimizer=keras.optimizers.Adam(3e-3), loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])
    else:
        m = keras.Sequential([layers.Input(shape=(X.shape[1],)), layers.Dense(32, activation="relu"), layers.Dense(1)])
        m.compile(optimizer=keras.optimizers.Adam(3e-3), loss="mse", metrics=["mae"])
    return m
# --- للصور: Input((H, W, C)) + [RandomTranslation/Flip] + Conv2D/MaxPool ×2 + Flatten + Dense + Dense(K, softmax) ; sparse_categorical_crossentropy
# --- للتسلسل: Input((W, k)) + GRU(16) + Dense(1) ; mse ; Adam(clipnorm=1.0)
model = build(); model.summary()

# [8] المعلمات الفائقة — جدول التجارب (أضف صفًا لكل تجربة على التحقق)
EXPERIMENTS = []   # [{"lr":3e-3, "hidden":32, "val_metric":...}, ...]

# [9] التدريب بحفظ نقاط
ckpt = os.path.join(OUT_DIR, "best.keras")
cbs = [callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True), callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True), callbacks.CSVLogger(os.path.join(OUT_DIR, "history.csv"))]
t0 = time.time()
h = model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=300, batch_size=32, callbacks=cbs, verbose=2)
print(f"trained {len(h.history['loss'])} epochs in {time.time() - t0:.0f}s on", "GPU" if tf.config.list_physical_devices("GPU") else "CPU")

# [10] التحقق: المنحنيات وأفضل حقبة
import matplotlib.pyplot as plt
plt.plot(h.history["loss"], label="loss"); plt.plot(h.history["val_loss"], label="val_loss"); plt.legend(); plt.xlabel("epoch"); plt.savefig(os.path.join(OUT_DIR, "curves.png")); plt.show()
print("best epoch:", int(np.argmin(h.history["val_loss"])) + 1)

# [11] التشخيص — اكتب في التقرير: قصور/فرط تخصيص/تسريب؟ ما فُعل؟

# [12] التقييم — مرة واحدة على الاختبار، مقابل خط الأساس، وعدة بذور إن أمكن
res = model.evaluate(X[te], y[te], verbose=0, return_dict=True); print("TEST:", {k: round(v, 4) for k, v in res.items()})
# --- للتصنيف: مصفوفة التباس وعتبة مختارة على التحقق ; للانحدار: اعكس توحيد الهدف وأبلغ بوحدته

# [13] التفسير — أهمية التبديل على التحقق
def perm_importance(model, Xv, yv, names, metric):
    base = metric(model.predict(Xv, verbose=0).ravel(), yv); out = []
    for j, n in enumerate(names):
        Xp = Xv.copy(); Xp[:, j] = np.random.default_rng(0).permutation(Xp[:, j]); out.append((n, metric(model.predict(Xp, verbose=0).ravel(), yv) - base))
    return sorted(out, key=lambda t: -abs(t[1]))
# مثال للانحدار: perm_importance(model, X[va], y[va], feature_names, lambda p, t: np.sqrt(np.mean((p - t) ** 2)))

# [14] الاستنساخ — احفظ كل ما يلزم لإعادة الأرقام
json.dump({"seed": SEED, "task": TASK, "target": TARGET, "splits": {"train": len(tr), "val": len(va), "test": len(te)}, "mu": mu.tolist(), "sd": sd.tolist(),
           "features": feature_names, "test_metrics": {k: float(v) for k, v in res.items()}, "versions": {"tensorflow": tf.__version__, "keras": keras.__version__}}, open(os.path.join(OUT_DIR, "run.json"), "w"), indent=2)
model.save(os.path.join(OUT_DIR, "final.keras"))
print("saved:", os.listdir(OUT_DIR))'''


def render() -> None:
    lesson_header(LESSON)
    why("القالب يمنع أكثر أخطاء المشاريع شيوعًا بالبنية نفسها: التقسيم قبل المعالجة، خط الأساس قبل النموذج، الاختبار مرة واحدة، والحفظ في كل خطوة. انسخه إلى دفتر Colab واستبدل ما بين `<>` وأسطر `# ---` بحسب نوع بياناتك.")
    st.code(TEMPLATE, language="python")
    h2("نقاط الاستبدال بحسب نوع المشروع", "Replacement points by project type")
    compare_table(["الخلية", "جدولي (كما في القالب)", "صور", "تسلسل"],
                  [("[2] التحميل", "read_csv", "تحميل صور إلى مصفوفة (n, H, W, C) + وسوم", "سلسلة/سلاسل بترتيب زمني"), ("[3] التشخيص", "مفقود/مكرر/توازن/ارتباط", "توازن الفئات، أحجام الصور، أمثلة لكل فئة", "رسم السلسلة، موسمية، قفزات، مفقود"),
                   ("[4]+[5] المعالجة والتقسيم", "توحيد + one-hot، تقسيم عشوائي مع stratify", "/255 + قناة، تقسيم عشوائي مع stratify", "توحيد بالماضي + نوافذ، تقسيم زمني"), ("[6] خط الأساس", "الأغلبية/المتوسط + لوجستي/خطي", "الأغلبية + لوجستي على البكسلات", "الساذج + المتوسط الموسمي"),
                   ("[7] البنية", "MLP صغير", "CNN صغيرة + زيادة بيانات", "GRU/LSTM + clipnorm"), ("[12] التقييم", "مصفوفة التباس/عتبة أو RMSE بوحدته", "مصفوفة التباس متعددة الفئات", "RMSE بوحدة السلسلة مقابل الساذج"), ("[13] التفسير", "أهمية التبديل", "خرائط الخصائص + أسوأ الصور", "أسوأ الفترات + أثر النافذة")],
                  ["rtl", "rtl", "rtl", "rtl"])
    practical_note("في Colab: ضع `drive.mount` في [0]، واجعل `OUT_DIR` داخل Drive حتى تبقى نقاط الحفظ بعد انتهاء الجلسة (الأسبوع 13). طوّر على عينة صغيرة (`df = df.sample(200)`) حتى يعمل الدفتر كاملًا، ثم أزل السطر.")
    intuition("الدفتر الجيد يقرؤه المقيّم كتقرير: عنوان الخلية = المرحلة، تعليق قصير = القرار والمبرر، مخرج مطبوع = الدليل. لا خلايا تجريبية متروكة ولا مخرجات قديمة.")
    with st.container(horizontal=True):
        st.button("دليل المشروع", icon=":material/checklist:", on_click=goto, args=("course.w14.project_guide",), key="w14_go_guide")
        st.button("الرمز النهائي والعرض (الأسبوع 15)", icon=":material/grading:", type="primary", on_click=goto, args=("course.w15",), key="w14_go_w15")
    common_mistake("تشغيل الدفتر خلية خلية بترتيب مختلف ثم تسليمه: المقيّم يشغّله من الأول فيفشل. قبل التسليم: Runtime → Restart and run all، ثم احفظ مع المخرجات.")
    post_test("course.w14", "14", [
        Q("أول قرار في المشروع:", ["البنية", "تعريف المسألة والهدف ووحدته", "المحسّن"], 1, ""),
        Q("خط الأساس…", ["اختياري", "إلزامي ويُبلَّغ بجانب النموذج", "للانحدار فقط"], 1, ""),
        Q("الاستنساخ يتطلب…", ["دقة عالية", "بذرة ونسخًا وكودًا يعيد الأرقام", "GPU"], 1, ""),
        Q("مجموعة الاختبار…", ["لضبط η", "تُلمس مرة واحدة", "تُخلط مع التحقق"], 1, ""),
        Q("الترتيب الصحيح:", ["معالجة ثم تقسيم", "تقسيم ثم معالجة بإحصاءات التدريب", "لا فرق"], 1, ""),
        Q("قبل التسليم:", ["احذف المخرجات", "Restart and run all ثم احفظ", "أرسل الخلايا المهمة فقط"], 1, ""),
        Q("جدول المعلمات الفائقة يُبنى على…", ["التدريب", "التحقق", "الاختبار"], 1, ""),
    ])
    takeaway("قالب واحد لثلاثة أنواع: 15 خلية تطابق المراحل. استبدل ما بين <> والأسطر المعلَّمة، طوّر على عينة، شغّل من الأول قبل التسليم، واحفظ إلى Drive.")
    lesson_footer(LESSON, ["القالب الكامل.", "نقاط الاستبدال.", "الاختبار البعدي."])
