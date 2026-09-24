import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from components.animation_player import Frame, animation_player, caption
from content.course.w11_lstm._viz import padding_svg
from core.models import Lesson

LESSON = Lesson(
    id="course.w11.data_prep_windows_padding",
    title_ar="إعداد البيانات التسلسلية: نوافذ متعددة المتغيرات، الأهداف متعددة الخطوات، والحشو للأطوال المختلفة",
    title_en="Sequence Data Preparation: Multivariate Windows, Multi-step Targets & Padding",
    module="course.w11",
    order=2,
    prerequisites=["course.w11.overview", "course.w10.sequences_hidden_state", "foundations.prep.tensors_batching_pipelines"],
    objectives_ar=["نوافذ بعدة متغيرات (features > 1) وأفق تنبؤ > 1 وأشكالها.", "الحشو والقناع للتسلسلات ذات الأطوال المختلفة (نصوص، معاملات عملاء) — بتحريك.", "تحجيم كل متغير بإحصاءات الماضي وتقسيم زمني."],
    terms=["sequence", "shape", "standardization", "padding", "timestep", "lstm"],
    difficulty="intermediate",
    summary_ar="متعدد المتغيرات: (samples, timesteps, 3). أفق h: y بشكل (samples, h). أطوال مختلفة: pad_sequences إلى طول موحّد + Masking لتجاهل الحشو. كل متغير يُوحَّد بإحصاءات التدريب.",
)

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, keras
from keras import layers
from labs.datasets import monthly_inflation

df = monthly_inflation()
S = df[["inflation", "policy_rate", "unemployment"]].to_numpy("float32")          # (180, 3): ثلاثة متغيرات شهرية
mu, sd = S[:120].mean(0), S[:120].std(0); Z = (S - mu) / sd                          # توحيد كل عمود بإحصاءات أول 120 شهرًا
# 1) نوافذ متعددة المتغيرات، هدف = التضخم بعد h أشهر
window, horizon = 12, 3
X = np.stack([Z[t - window:t] for t in range(window, len(Z) - horizon + 1)])          # (n, 12, 3)
y = np.stack([Z[t:t + horizon, 0] for t in range(window, len(Z) - horizon + 1)])      # (n, 3): التضخم للأشهر الثلاثة القادمة
print("multivariate windows X:", X.shape, "(samples, timesteps, features) | multi-step y:", y.shape, "(samples, horizon)")
print("X[0, -1] = last month of window 0 (3 variables):", X[0, -1].round(2), "| y[0] = next 3 inflation values:", y[0].round(2))
k = 108; print("time split -> train windows:", k, "| test windows:", len(X) - k, "(later in time, no shuffle)")

# 2) أطوال مختلفة (مثلًا سجل معاملات لكل عميل): حشو + قناع
seqs = [[0.5, 1.2, -0.3], [2.0], [0.1, 0.4, 0.9, 1.3, -1.0]]                          # ثلاث سلاسل بأطوال 3، 1، 5
padded = keras.utils.pad_sequences(seqs, padding="post", dtype="float32", value=0.0)   # (3, 5): أصفار في النهاية
print("\\npadded:\\n", padded)
X_pad = padded[..., None]                                                            # (3, 5, 1)
model = keras.Sequential([layers.Input(shape=(5, 1)), layers.Masking(mask_value=0.0), layers.LSTM(4), layers.Dense(1)])
out_masked = model.predict(X_pad, verbose=0).ravel()
# بدون قناع: الأصفار تُقرأ كمدخلات حقيقية
model2 = keras.Sequential([layers.Input(shape=(5, 1)), layers.LSTM(4), layers.Dense(1)]); model2.set_weights(model.get_weights())
print("with Masking vs without, output for sequence #2 (length 1):", out_masked[1].round(4), "vs", model2.predict(X_pad, verbose=0).ravel()[1].round(4), "<- padding changes the answer unless masked")
print("model params:", model.count_params(), "= 4 * ((1 + 4) * 4 + 4) + (4 + 1)")'''


def render() -> None:
    lesson_header(LESSON)
    why("الأسبوع 10 استخدم متغيرًا واحدًا وأفقًا واحدًا وأطوالًا موحّدة. الواقع: عدة متغيرات (تضخم، فائدة، بطالة)، أفق عدة أشهر، وسلاسل بأطوال مختلفة (كل عميل له عدد معاملات مختلف). الأشكال تتغير — والأخطاء الصامتة تبدأ هنا.")
    definition("**النافذة متعددة المتغيرات**: كل خطوة زمنية متجه features. **الهدف متعدد الخطوات**: y بطول horizon. **الحشو** `padding`: إكمال التسلسلات القصيرة بقيمة خاصة إلى طول موحّد كي تُجمَّع في موتر واحد، مع **قناع** `Masking` يخبر الطبقة التكرارية بتجاهل الخطوات المحشوّة.")
    pcaps = ["**أربعة عملاء، أربعة أطوال**: عدد المعاملات الشهرية يختلف (3، 6، 2، 4). لا يمكن رصّها في موتر واحد بشكل (n, T, k) لأن T مختلف.",
             "**الحشو اللاحق** `padding='post'`: نكمل كل تسلسل بأصفار (رمادي متقطع) حتى أطول طول. الآن الموتر (4, 7, 1) — لكن الأصفار ستُقرأ كقيم حقيقية!",
             "**القناع** `Masking(mask_value=0)`: قناع 1/0 لكل خطوة يخبر LSTM بتخطي الخطوات المحشوّة: تُنقل الحالة كما هي دون تحديث. النتيجة لا تتأثر بطول الحشو."]
    animation_player("w11_pad", [Frame(padding_svg(i), caption(c), action=["lengths", "pad", "mask"][i]) for i, c in enumerate(pcaps)],
                     title_ar="من تسلسلات بأطوال مختلفة إلى موتر واحد", interval_ms=2600)
    code_lab(CodeLab(
        key="w11_prep", title_ar="نوافذ بثلاثة متغيرات وأفق ثلاثة أشهر، ثم حشو وقناع لأطوال مختلفة", code=CODE, level="B",
        before=Before(goal_ar="بناء X (n, 12, 3) وy (n, 3) من ثلاث سلاسل شهرية بتوحيد صحيح وتقسيم زمني، ثم حشو ثلاث سلاسل بأطوال مختلفة وإثبات أن القناع يغيّر الناتج.", stage_ar="الأسبوع 11: إعداد البيانات.",
                      inputs_ar="monthly_inflation (3 أعمدة)، وثلاث سلاسل يدوية.", expected_ar="X (166, 12, 3)، y (166, 3)؛ مصفوفة محشوّة (3, 5) بأصفار في النهاية؛ ناتج مختلف مع/بدون Masking للسلسلة القصيرة؛ 101 معلمة."),
        explain=[("6-8", "ثلاثة متغيرات؛ التوحيد لكل عمود بإحصاءات الماضي (أول 120 شهرًا) — لا الكل."), ("10-13", "النافذة تأخذ الأعمدة الثلاثة؛ الهدف الأشهر الثلاثة القادمة من التضخم فقط: y بشكل (n, 3) → Dense(3) في المخرج."), ("15", "تقسيم زمني بالفهرس؛ لا خلط."),
                 ("18-20", "`pad_sequences` يكمل بالأصفار (post = في النهاية) ويعطي مصفوفة واحدة."), ("21-26", "`Masking(mask_value=0)` يجعل LSTM يتخطى خطوات الحشو. بلا قناع تُقرأ الأصفار كقيم حقيقية فيتغير الناتج."), ("27", "معلمات LSTM = 4 بوابات × ((features + units) × units + units).")],
        run=run_printed(CODE),
        after_ar="- (n, 12, 3): آخر بُعد هو عدد المتغيرات؛ الأسبوع 10 كان 1.\n- y (n, 3): انحدار متعدد المخرجات — الخسارة MSE على الثلاثة معًا.\n- القناع يعمل فقط إذا كانت قيمة الحشو لا تظهر في البيانات الحقيقية (0 بعد التوحيد قد يظهر! استخدم قيمة مثل −99 أو حشوًا قبل التوحيد بحذر).",
    ))
    h2("أشكال الإعداد الشائعة", "Common preparation shapes")
    compare_table(["المسألة", "X", "y", "المخرج", "الخسارة"],
                  [("قيمة تالية، متغير واحد", "(n, T, 1)", "(n,)", "Dense(1)", "mse"), ("قيمة تالية، k متغيرات", "(n, T, k)", "(n,)", "Dense(1)", "mse"), ("h خطوات قادمة", "(n, T, k)", "(n, h)", "Dense(h)", "mse"),
                   ("تصنيف تسلسل (سيتعثر العميل؟)", "(n, T_max, k) + قناع", "(n,)", "Dense(1, sigmoid)", "binary_crossentropy"), ("نص → فئة", "(n, T_max) أعداد صحيحة → Embedding", "(n,)", "Dense(K, softmax)", "sparse_categorical_crossentropy"), ("وسم لكل خطوة", "(n, T, k)", "(n, T)", "LSTM(return_sequences=True) + Dense", "بحسب الهدف")],
                  ["rtl", "code", "code", "code", "code"])
    intuition("سؤالان لكل مسألة تسلسل: «ما شكل الملاحظة الواحدة؟» (T × k) و«ما شكل الجواب؟» (رقم، متجه، أو تسلسل). الجوابان يحددان النافذة والرأس والخسارة — الباقي هو Keras.")
    common_mistake("توحيد كل متغير بإحصاءات السلسلة كاملة، أو نسيان توحيد متغير بمقياس مختلف (الفائدة بالمئات بجانب التضخم بالوحدات): LSTM تتشبع على المتغير الكبير. كل عمود بمتوسطه وانحرافه من فترة التدريب.")
    quiz("w11.prep", [
        Q("نافذة 24 خطوة بخمسة متغيرات وأفق 6: أشكال X وy", ["(n, 24, 5) و(n, 6)", "(n, 120) و(n,)", "(n, 5, 24) و(n, 6)"], 0, ""),
        Q("Masking يفيد عندما…", ["كل السلاسل بنفس الطول", "السلاسل محشوّة إلى طول موحّد", "البيانات موحَّدة"], 1, ""),
        Q("LSTM(8) على features=3: المعلمات", ["96", "384", "352"], 1, "4 × ((3+8)×8 + 8)."),
        Q("قيمة الحشو يجب أن…", ["تكون 0 دائمًا", "لا تظهر كقيمة حقيقية في البيانات", "تساوي المتوسط"], 1, ""),
    ])
    takeaway("X (n, T, k)، y بحسب الهدف (رقم/متجه/تسلسل). كل متغير يُوحَّد بإحصاءات الماضي. أطوال مختلفة → حشو + قناع بقيمة لا تظهر في البيانات. التقسيم زمني.")
    lesson_footer(LESSON, ["متعدد المتغيرات ومتعدد الخطوات بالكود.", "الحشو والقناع بالدليل.", "جدول الأشكال الشائعة."])
