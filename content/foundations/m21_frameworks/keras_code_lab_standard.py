import numpy as np
import streamlit as st

from components.callouts import common_mistake, debugging_note, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.diagram import diagram, svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.frameworks.keras.code_lab_standard",
    title_ar="معيار معمل كود Keras: مثال كامل بكل بنود العقد",
    title_en="The Keras Code Lab Standard: A Complete Example",
    module="foundations.frameworks",
    parent="foundations.frameworks.keras",
    order=12,
    prerequisites=["foundations.frameworks.keras.errors", "foundations.prep.scaling", "foundations.prep.encoding", "foundations.eval.classification_metrics"],
    objectives_ar=["تطبيق «عقد فهم الكود» كاملًا على مثال Keras: 11 سؤالًا قبل الكود، الكود، سطرًا سطرًا، التشغيل، تفسير المخرجات، التشخيص.", "بناء خط أنابيب كامل: بيانات جدولية ← تحجيم/ترميز ← MLP ← تدريب مع إيقاف مبكر ← تقييم ← مقارنة بخط أساس.", "استخدام هذا القالب لكل مثال Keras لاحق في المقرر."],
    terms=["standardization", "categorical", "cross_entropy", "epoch"],
    labs=["labs.confusion_matrix_lab", "labs.curves_diagnostic_lab"],
    difficulty="intermediate",
    summary_ar="قبل أي كود Keras أجب: لماذا Keras؟ ما المدخل؟ ما الهدف؟ البنية والأشكال؟ لماذا هذا التنشيط/الخسارة/المقياس/المحسّن/حجم الدفعة/عدد الحقب؟ ثم الكود والتشغيل والتفسير والتشخيص.",
)

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, keras
from keras import layers, callbacks
from labs.datasets import loan_default
keras.utils.set_random_seed(0)

# 1) البيانات: تعثر القروض (وحدة 1)، 400 عميل، هدف ثنائي defaulted
df = loan_default(n=400, seed=7)
num_cols = ["income", "age", "num_late_payments", "debt_ratio"]
cat_cols = ["city", "employment"]
X_num = df[num_cols].to_numpy("float32")
X_cat = pd.get_dummies(df[cat_cols]).to_numpy("float32")          # one-hot (وحدة 8): 4 مدن + 3 حالات = 7 أعمدة
y = df["defaulted"].to_numpy("float32")

# 2) التقسيم قبل أي تحجيم (وحدة 8: لا تسريب)
rng = np.random.default_rng(0); idx = rng.permutation(len(df))
tr, va, te = idx[:260], idx[260:330], idx[330:]
med = np.nanmedian(X_num[tr], axis=0)                              # قيم مفقودة في debt_ratio: وسيط التدريب فقط
X_num = np.where(np.isnan(X_num), med, X_num)
mu, sd = X_num[tr].mean(0), X_num[tr].std(0) + 1e-8                # إحصاءات التدريب فقط
X = np.concatenate([(X_num - mu) / sd, X_cat], axis=1)             # (400, 11)
X_tr, y_tr, X_va, y_va, X_te, y_te = X[tr], y[tr], X[va], y[va], X[te], y[te]
print("shapes:", X_tr.shape, X_va.shape, X_te.shape, "| default rate train:", y_tr.mean().round(3))

# 3) خط الأساس (وحدة 7): التنبؤ بالفئة الغالبة
baseline_acc = max(y_te.mean(), 1 - y_te.mean())
print(f"baseline accuracy (majority class): {baseline_acc:.3f}")

# 4) النموذج
model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),                # (None, 11)
    layers.Dense(16, activation="relu"),              # (None, 16): 11*16+16 = 192
    layers.Dropout(0.2),                              # تنظيم (وحدة 20)
    layers.Dense(8, activation="relu"),               # (None, 8): 16*8+8 = 136
    layers.Dense(1, activation="sigmoid"),            # (None, 1): 8+1 = 9 -> احتمال التعثر
], name="loan_mlp")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-3),
              loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.Recall(name="recall")])

# 5) التدريب مع إيقاف مبكر على التحقق
es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
h = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=200, batch_size=32, verbose=0, callbacks=[es])
best = int(np.argmin(h.history["val_loss"])) + 1
print(f"trained {len(h.history['loss'])} epochs (best val_loss {min(h.history['val_loss']):.4f} at epoch {best})")
print("params:", model.count_params())

# 6) التقييم على الاختبار (مرة واحدة)
loss, acc, auc, rec = model.evaluate(X_te, y_te, verbose=0)
print(f"test: loss={loss:.4f} accuracy={acc:.3f} auc={auc:.3f} recall@0.5={rec:.3f}")
p = model.predict(X_te, verbose=0).ravel()
for thr in (0.3, 0.5):
    pred = (p >= thr).astype(int); tp = ((pred == 1) & (y_te == 1)).sum(); fn = ((pred == 0) & (y_te == 1)).sum(); fp = ((pred == 1) & (y_te == 0)).sum()
    print(f"  threshold {thr}: recall={tp / max(tp + fn, 1):.3f} precision={tp / max(tp + fp, 1):.3f} flagged={pred.sum()}/{len(pred)}")'''


def _arch_svg() -> str:
    boxes = [("Input", "(None, 11)", "#E6F1FB"), ("Dense 16 · relu", "(None, 16)", "#E3F3F0"), ("Dropout 0.2", "(None, 16)", "#F1EFEA"), ("Dense 8 · relu", "(None, 8)", "#E3F3F0"), ("Dense 1 · sigmoid", "(None, 1)", "#FBE6E2")]
    s = '<svg viewBox="0 0 760 130" width="100%" style="max-width:760px">' + svg_defs()
    x = 10
    for i, (lbl, shp, fill) in enumerate(boxes):
        s += svg_box(x, 30, 135, 50, lbl, fill, font=12, bold=True) + svg_text(x + 67, 105, shp, size=11, mono=True, color="#6B675F")
        if i < 4:
            s += svg_arrow(x + 137, 55, x + 150, 55)
        x += 152
    s += svg_text(380, 18, "params: 192 + 0 + 136 + 9 = 337", size=11, mono=True, color="#6B675F")
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    why("«عقد فهم الكود» (المقدمة) يمنع نسخ القوالب بلا فهم. لكل مثال Keras في المقرر نجيب عن **11 سؤالًا قبل الكود**. هذه الصفحة النموذج الكامل؛ ما بعدها يشير إليها.")
    h2("قبل الكود: الأسئلة الأحد عشر", "Before the code: the eleven questions")
    compare_table(["السؤال", "الجواب لهذا المثال"],
                  [("لماذا Keras هنا؟", "مسألة جدولية قياسية بـ MLP صغير: لا حاجة لحلقة صريحة؛ نريد الإيقاف المبكر والمقاييس جاهزة."),
                   ("ما مدخل النموذج؟", "11 خاصية لكل عميل: 4 عددية مُوحَّدة (إحصاءات التدريب) + 7 one-hot (مدينة، حالة العمل). شكل الدفعة (batch, 11)."),
                   ("ما الهدف؟", "`defaulted` ∈ {0, 1}: تعثر (1) أم لا (0). نسبة الإيجابيات ≈ 46% في التدريب (غير متوازن قليلًا)."),
                   ("البنية؟", "انظر الرسم: 11 → 16 (ReLU) → Dropout → 8 (ReLU) → 1 (sigmoid)."),
                   ("الأشكال عند كل طبقة؟", "(None, 11) → (None, 16) → (None, 16) → (None, 8) → (None, 1)."),
                   ("لماذا هذا التنشيط؟", "ReLU في الوسط (افتراضي آمن، وحدة 11)؛ sigmoid في المخرج لاحتمال ثنائي (وحدة 12)."),
                   ("لماذا هذه الخسارة؟", "binary_crossentropy: هدف ثنائي + sigmoid (وحدة 13، جدول التوافق)."),
                   ("لماذا هذه المقاييس؟", "accuracy للمقارنة بخط الأساس؛ AUC لجودة الترتيب مستقلًا عن العتبة؛ recall لأن تفويت متعثر أغلى من إنذار كاذب (وحدة 18)."),
                   ("لماذا هذا المحسّن؟", "Adam بـ η = 3e-3: افتراضي قوي لمسألة صغيرة (وحدة 15)."),
                   ("لماذا هذا حجم الدفعة؟", "32: 260/32 ≈ 9 تحديثات لكل حقبة؛ صغير كفاية للضوضاء المفيدة وكبير كفاية للاستقرار (وحدة 17)."),
                   ("لماذا هذا عدد الحقب؟", "200 حد أعلى فقط؛ الإيقاف المبكر (patience=10) يقرر الفعلي (وحدة 20).")],
                  ["rtl", "rtl"])
    diagram("بنية النموذج والأشكال", _arch_svg(), what_ar="خمس طبقات مع شكل مخرج كل واحدة وعدد المعلمات الكلي.", how_ar="تحقق: 11×16+16 = 192، 16×8+8 = 136، 8×1+1 = 9. Dropout وInput بلا معلمات.", takeaway_ar="337 معلمة لـ 260 ملاحظة تدريب: نسبة تستدعي التنظيم (Dropout) والإيقاف المبكر.", title_en="Architecture")
    code_lab(CodeLab(
        key="keras_standard", title_ar="خط أنابيب كامل: تعثر القروض بـ Keras", code=CODE, level="C",
        before=Before(goal_ar="من DataFrame خام إلى نموذج مقيَّم على اختبار محجوز، مع خط أساس وعتبتين.", stage_ar="خط الأنابيب الكامل (المستوى C).",
                      prerequisites_ar="وحدات 8 (تقسيم، تحجيم، ترميز)، 13، 15، 18، 20.", inputs_ar="loan_default(400): 4 أعمدة عددية + 2 فئوية + هدف ثنائي.",
                      expected_ar="أشكال (260, 11)/(70, 11)/(70, 11)؛ خط أساس ≈ 0.59؛ توقف قبل 200 حقبة؛ test accuracy > خط الأساس وAUC > 0.7؛ العتبة 0.3 ترفع الاستدعاء وتخفض الصحة.",
                      objects_ar="`Sequential`, `Dense`, `Dropout`, `Adam`, `AUC`, `Recall`, `EarlyStopping`, `History`."),
        explain=[("7-13", "قراءة البيانات وفصل العددي عن الفئوي. `get_dummies` = one-hot (وحدة 8). الهدف float32 للخسارة الثنائية."),
                 ("15-22", "التقسيم **قبل** أي إحصاء: الوسيط لملء المفقود، ثم المتوسط/الانحراف — كلها من التدريب فقط ثم تُطبَّق على الكل (منع التسريب، وحدة 8). دمج العددي المحجّم مع one-hot."),
                 ("25-26", "خط الأساس (وحدة 7): إن لم يتفوق النموذج عليه فلا قيمة له."), ("29-36", "البنية بالأشكال والمعلمات في التعليقات. Dropout بعد أول طبقة مخفية."),
                 ("37-39", "compile: Adam بـ η صريح؛ الخسارة الثنائية؛ ثلاثة مقاييس بأسماء واضحة (تظهر في History كـ `auc`, `val_auc`…)."),
                 ("41-45", "الإيقاف المبكر مع استرجاع أفضل أوزان. epochs=200 حد أعلى. `best` من History."), ("48-53", "التقييم مرة واحدة على الاختبار؛ ثم predict وعتبتان: 0.3 لالتقاط متعثرين أكثر بثمن إنذارات كاذبة (وحدة 18).")],
        run=run_printed(CODE),
        after_ar="""- **الأشكال**: 11 = 4 + 7. لو رأيت 10 أو 12 فالترميز تغيّر (فئة مفقودة في العينة).
- **trained N epochs**: N ≪ 200 = الإيقاف المبكر عمل؛ best هي الحقبة المسترجَعة.
- **test accuracy مقابل baseline**: الفرق هو القيمة الحقيقية. إن كان صغيرًا فالخصائص ضعيفة أو النموذج غير مناسب — لا تزد الطبقات بلا سبب.
- **AUC** يقيّم الترتيب بلا عتبة؛ **recall@0.5** يعتمد عليها. سطرا العتبة يوضحان المقايضة: 0.3 يرفع الاستدعاء (يلتقط متعثرين أكثر) ويخفض الصحة (إنذارات أكثر). القرار للجهة المستخدمة (وحدة 18).
- كل ما يُحفظ للاستخدام لاحقًا: النموذج + `mu, sd` + أعمدة one-hot بترتيبها + العتبة المختارة.""",
    ))
    h2("التشخيص", "Diagnostics")
    debugging_note("**إن كانت accuracy = خط الأساس بالضبط**: النموذج يتنبأ بفئة واحدة. تحقق من: تحجيم المدخل (income بآلاف بلا توحيد يقتل ReLU)، η، وأن y ليس كله صفرًا في دفعة التدريب. **إن كان val_loss يرتفع من الحقبة 3**: زد Dropout أو قلّل الوحدات أو زد البيانات. **إن ظهرت nan**: قلّل η وتحقق من `debt_ratio` (قسمة على صفر في التحجيم إن كان sd = 0).")
    common_mistake("تحجيم البيانات كلها قبل التقسيم، أو `get_dummies` على التدريب والاختبار منفصلين (أعمدة مختلفة إن غابت فئة). رمّز على الكل ثم قسّم، لكن **حجّم** بإحصاءات التدريب فقط.")
    intuition("هذا القالب هو ما ستكرره في الأسابيع 2–7: بيانات ← تقسيم ← تحجيم/ترميز ← خط أساس ← نموذج ← إيقاف مبكر ← تقييم واحد على الاختبار ← عتبة بحسب الكلفة. تغيير المسألة يغيّر الأجوبة عن الأسئلة الأحد عشر، لا الهيكل.")
    quiz("keras.standard", [
        Q("متى تُحسب إحصاءات التحجيم؟", ["على كل البيانات", "على التدريب فقط بعد التقسيم", "على الاختبار"], 1, "منع التسريب."),
        Q("لماذا recall مقياسًا هنا؟", ["لأنه الافتراضي", "لأن تفويت متعثر أغلى من إنذار كاذب", "لأنه قابل للاشتقاق"], 1, "الكلفة."),
        Q("`epochs=200` مع EarlyStopping يعني…", ["200 حقبة دائمًا", "حد أعلى؛ الفعلي يقرره التحقق", "200 دفعة"], 1, "حد."),
        Q("accuracy = خط الأساس بالضبط يشير إلى…", ["نموذج ممتاز", "تنبؤ بفئة واحدة", "تسريب"], 1, "لا تعلّم."),
    ])
    takeaway("11 سؤالًا قبل الكود → الكود → سطرًا سطرًا → تشغيل → تفسير → تشخيص. هذا هو معيار كل معمل Keras في المقرر.")
    lesson_footer(LESSON, ["الأسئلة الأحد عشر.", "خط أنابيب كامل بلا تسريب.", "التقييم مرة واحدة والعتبة بالكلفة."])
