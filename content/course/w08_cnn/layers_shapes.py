import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, intuition, math_note, research_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.diagram import diagram, svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.quiz import Q
from components.week import post_test
from core.models import Lesson
from content.course.w08_cnn._viz import flow_svg, receptive_svg, shape_rows
from core.routing import go as goto
from core.rtl import table
from labs.cnn import shape_trace

LESSON = Lesson(
    id="course.w08.layers_shapes",
    title_ar="طبقات CNN الأساسية، البنى النموذجية، وحساب شكل المخرج + الاختبار البعدي",
    title_en="Core CNN Layers, Typical Architectures & Output-Shape Calculation — & Post-test",
    module="course.w08",
    order=4,
    prerequisites=["course.w08.padding_stride_pooling", "foundations.architecture.parameter_count", "foundations.frameworks.keras.summary_deconstruction"],
    objectives_ar=["الطبقات: Conv2D، Activation، MaxPooling، Flatten/GlobalAveragePooling، Dense، Dropout/BN.", "البنية النموذجية (LeNet-style) وتتبع الأشكال والمعلمات يدويًا (تحريك القمع) ثم في summary.", "نمو مجال الرؤية `receptive field` مع العمق (تحريك).", "أين تتركز المعلمات ولماذا، والاختبار البعدي."],
    terms=["parameter", "shape", "cnn", "convolution", "pooling", "feature_map", "dense_layer"],
    labs=["labs.cnn_shape_calculator"],
    difficulty="intermediate",
    summary_ar="[Conv → ReLU → Pool] × n → Flatten → Dense → softmax. Conv: k²·C_in·C_out + C_out؛ Dense بعد Flatten: H·W·C × units — الأثقل. احسب الأشكال بالصيغة وطابق summary.",
)

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras
from keras import layers
from labs.cnn import shape_trace

spec = [{"type": "conv", "k": 3, "filters": 16, "padding": "same"}, {"type": "pool", "size": 2},
        {"type": "conv", "k": 3, "filters": 32, "padding": "same"}, {"type": "pool", "size": 2},
        {"type": "flatten"}, {"type": "dense", "units": 64}, {"type": "dense", "units": 10}]
print("hand-traced shapes (28×28×1 input):")
total = 0
for r in shape_trace(28, 1, spec):
    total += r["params"]; print(f"  {r['layer']:<38} {r['shape']:<14} params {r['params']:>7,}")
print("  hand total:", f"{total:,}")

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, 3, padding="same", activation="relu"),   # 3*3*1*16 + 16 = 160
    layers.MaxPooling2D(2),                                    # 28 -> 14
    layers.Conv2D(32, 3, padding="same", activation="relu"),   # 3*3*16*32 + 32 = 4,640
    layers.MaxPooling2D(2),                                    # 14 -> 7
    layers.Flatten(),                                          # 7*7*32 = 1,568
    layers.Dense(64, activation="relu"),                       # 1568*64 + 64 = 100,416  <- الأثقل
    layers.Dense(10, activation="softmax"),                    # 64*10 + 10 = 650
])
model.summary()
print("Keras total:", f"{model.count_params():,}", "| match:", model.count_params() == total)
# البديل: GlobalAveragePooling بدل Flatten يقلّص Dense
gap = keras.Sequential([layers.Input(shape=(28, 28, 1)), layers.Conv2D(16, 3, padding="same", activation="relu"), layers.MaxPooling2D(2), layers.Conv2D(32, 3, padding="same", activation="relu"), layers.GlobalAveragePooling2D(), layers.Dense(10, activation="softmax")])
print("with GlobalAveragePooling2D instead of Flatten+Dense(64):", f"{gap.count_params():,}", "params")'''


def _arch_svg() -> str:
    boxes = [("Input", "28×28×1", "#E6F1FB"), ("Conv 16, 3×3 same\n+ ReLU", "28×28×16", "#E3F3F0"), ("MaxPool 2", "14×14×16", "#F1EFEA"), ("Conv 32, 3×3 same\n+ ReLU", "14×14×32", "#E3F3F0"), ("MaxPool 2", "7×7×32", "#F1EFEA"), ("Flatten", "1568", "#FFF3D6"), ("Dense 64 + ReLU", "64", "#EFE9F8"), ("Dense 10 softmax", "10", "#FBE6E2")]
    s = '<svg viewBox="0 0 900 150" width="100%" style="max-width:900px">' + svg_defs()
    x = 8
    for i, (lbl, shp, fill) in enumerate(boxes):
        if "\n" in lbl:
            a, b = lbl.split("\n"); s += svg_box(x, 30, 104, 56, "", fill) + svg_text(x + 52, 52, a, size=10, bold=True) + svg_text(x + 52, 68, b, size=10, bold=True)
        else:
            s += svg_box(x, 30, 104, 56, lbl, fill, font=11, bold=True)
        s += svg_text(x + 52, 110, shp, size=10, mono=True, color="#6B675F")
        if i < len(boxes) - 1:
            s += svg_arrow(x + 106, 58, x + 112, 58)
        x += 112
    s += svg_text(450, 140, "(H, W) shrink 28 → 14 → 7 while C grows 1 → 16 → 32; then the classifier head", size=11, color="#6B675F")
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("الطبقات الأساسية", "The core layers")
    compare_table(["الطبقة", "الدور", "الشكل", "المعلمات", "Keras"],
                  [("Conv2D", "كواشف أنماط محلية متعلَّمة", "(H, W, C_in) → (H', W', filters)", "k²·C_in·filters + filters", "`Conv2D(32, 3, padding='same', activation='relu')`"), ("Activation (ReLU)", "لاخطية بعد كل التفاف", "ثابت", "0", "داخل Conv2D أو `Activation('relu')`"),
                   ("MaxPooling2D", "تقليص + ثبات للإزاحة", "(H, W, C) → (H/2, W/2, C)", "0", "`MaxPooling2D(2)`"), ("BatchNormalization", "تثبيت التدريب العميق", "ثابت", "4·C (2 متعلَّمة)", "`BatchNormalization()` (الأسس 20)"),
                   ("Dropout", "تنظيم (غالبًا في الرأس)", "ثابت", "0", "`Dropout(0.5)`"), ("Flatten", "من (H, W, C) إلى متجه", "→ (H·W·C,)", "0", "`Flatten()`"), ("GlobalAveragePooling2D", "متوسط كل خريطة → متجه C", "→ (C,)", "0", "`GlobalAveragePooling2D()`"),
                   ("Dense", "الرأس المصنّف/المنحدر", "→ (units,)", "in·units + units", "`Dense(10, activation='softmax')`")],
                  ["code", "rtl", "code", "code", "code"])
    h2("البنية النموذجية", "A typical architecture")
    diagram("CNN بأسلوب LeNet لصور 28×28", _arch_svg(), what_ar="كتلتان [Conv → ReLU → Pool] ثم رأس كثيف. تحت كل صندوق شكل مخرجه.", how_ar="تتبع (H, W, C): same يحفظ H وW، Pool ينصفهما، filters يحدد C. Flatten يضرب الثلاثة. Dense بعد Flatten تحمل معظم المعلمات.", takeaway_ar="القمع: مكاني ينكمش، قنوات تتسع، ثم مصنّف.", title_en="LeNet-style CNN")
    h3("القمع طبقةً طبقة", "The funnel, layer by layer")
    rows = shape_rows()
    fcaps = []
    for i, r in enumerate(rows):
        extra = {0: "صورة رمادية 28×28 بقناة واحدة.",
                 1: "16 نواة 3×3 بحشو same: الحجم المكاني ثابت، والقنوات 1 → 16 (16 كاشفًا). معلمات: 3·3·1·16 + 16 = 160.",
                 2: "التجميع ينصف H وW بلا معلمات: 28 → 14.",
                 3: "32 نواة، كل واحدة ترى **16 قناة**: 3·3·16·32 + 32 = 4,640 معلمة.",
                 4: "تجميع ثانٍ: 14 → 7. كل خلية الآن تلخّص منطقة كبيرة من الصورة الأصلية.",
                 5: "Flatten: 7 × 7 × 32 = 1,568 رقمًا في متجه واحد — نهاية الجزء المكاني.",
                 6: "Dense(64) على 1,568 مدخلًا: 1,568 × 64 + 64 = **100,416 معلمة** — 95% من الشبكة كلها في هذه الطبقة.",
                 7: "Dense(10) + softmax: احتمال لكل من 10 فئات. 64 × 10 + 10 = 650."}.get(i, "")
        fcaps.append(f"**{r['layer']}** → `{r['shape']}` · {r['params']:,} معلمة. {extra}")
    animation_player("w08_flow", [Frame(flow_svg(rows, i), caption(c), action=rows[i]["layer"].split("(")[0], values=[("params (layer)", "", f"{rows[i]['params']:,}"), ("params (total)", "", f"{sum(x['params'] for x in rows[:i + 1]):,}")])
                                  for i, c in enumerate(fcaps)], title_ar="الأشكال تنكمش مكانيًا وتتسع قنواتيًا", interval_ms=2200)
    h3("مجال الرؤية: كم يرى كل بكسل في العمق؟", "Receptive field: how much does a deep unit see?")
    animation_player("w08_rf", [Frame(receptive_svg(n), caption(f"**بعد {n} طبقة 3×3**: خلية واحدة في خريطة الخصائص تتأثر بنافذة {2 * n + 1}×{2 * n + 1} من الصورة الأصلية."
                                                           + (" الطبقة الأولى ترى 3×3 فقط — حواف صغيرة." if n == 1 else "")
                                                           + (" كل طبقة تضيف بكسلًا من كل جانب: هكذا تركّب الطبقات العميقة أنماطًا أكبر من أنماط أصغر." if n == 3 else "")),
                                     action=f"{n} layers") for n in range(1, 6)], title_ar="نمو مجال الرؤية بتكديس نوى 3×3", interval_ms=1600)
    math_note("لطبقات 3×3 بخطوة 1: RF = 1 + 2L. التجميع 2×2 أو الخطوة 2 **يضاعف** معدل النمو لكل ما بعدها، لذلك تصل الشبكات بعد بضع كتل إلى رؤية الصورة كلها بطبقات قليلة وأنوية صغيرة.")
    code_lab(CodeLab(
        key="w08_shapes", title_ar="تتبع الأشكال والمعلمات يدويًا ومطابقتها بـ summary، وبديل GlobalAveragePooling", code=CODE, level="B",
        before=Before(goal_ar="حساب شكل كل طبقة ومعلماتها بالصيغ ثم بناء نفس الشبكة في Keras والتحقق من التطابق، ورؤية أثر استبدال Flatten بـ GlobalAveragePooling.", stage_ar="الأسبوع 08: الأشكال.",
                      inputs_ar="مدخل 28×28×1 (حجم MNIST).", expected_ar="جدول يدوي = summary: 160 + 4,640 + 100,416 + 650 = 105,866؛ نسخة GAP بنحو 5,130 معلمة."),
        explain=[("6-13", "`shape_trace` (محرك المنصة) يطبّق صيغة الحجم وصيغ المعلمات طبقةً طبقة."), ("15-25", "نفس البنية في Keras بتعليق المعلمات على كل سطر. `summary()` يطبع الأشكال بـ None للدفعة."), ("26", "التطابق: لا فرق بين الحساب اليدوي وKeras."), ("28-29", "GlobalAveragePooling يحوّل 7×7×32 إلى 32 رقمًا: Dense(10) تحتاج 330 معلمة بدل 100 ألف.")],
        run=run_printed(CODE),
        after_ar="- **95% من المعلمات في Dense(64) بعد Flatten**: هذا نمط عام؛ لذلك تُقلَّص الصورة قبل Flatten أو يُستخدم GAP.\n- طبقات Conv رخيصة معلماتيًا (160، 4,640) لكنها الأغلى **حسابيًا** (تُطبَّق على كل موضع).\n- `(None, 28, 28, 16)` في summary: الدفعة None، ثم H، W، C.",
    ))
    st.button("افتح حاسبة أشكال CNN", icon=":material/science:", type="primary", on_click=goto, args=("labs.cnn_shape_calculator",), key="w08_lab_calc")
    intuition("اقرأ أي CNN كسؤالين متكررين: «كم انكمشت (H, W)؟» و«كم اتسعت C؟». الجواب الأول من الصيغة وPool، والثاني من filters. ثم سؤال واحد للرأس: «كم طول المتجه قبل Dense؟».")
    research_note("البنى الشهيرة (LeNet، AlexNet، VGG، ResNet) كلها هذا القالب بعمق أكبر وحيل (كتل متبقية، BN). ResNet مثلًا: [Conv 3×3 same + BN + ReLU] × كثير مع اتصالات قفز؛ الأشكال تُحسب بنفس الصيغ. في مشروع الأسبوع 09 نبقى في القالب الصغير.")
    common_mistake("Flatten مباشرة على 224×224×64 ثم Dense(256): 3.2 مليون × 256 ≈ 820 مليون معلمة. قلّص أولًا (Pool/stride) أو استخدم GAP. summary يكشف ذلك فورًا — اقرأه قبل fit.")
    post_test("course.w08", "08", [
        Q("صورة ملونة 32×32 كموتر (Keras):", ["(32, 32)", "(32, 32, 3)", "(3, 32)"], 1, ""),
        Q("نواة 3×3 على قناة واحدة: معلماتها", ["9", "10", "3"], 1, ""),
        Q("الالتفاف ينتج…", ["رقمًا واحدًا", "خريطة خصائص", "متجه أوزان"], 1, ""),
        Q("MaxPooling(2) على 28×28 يعطي", ["28×28", "14×14", "26×26"], 1, ""),
        Q("Conv2D(32, 3) بعد طبقة بـ 16 قناة: المعلمات", ["4,640", "288", "160"], 0, "3·3·16·32 + 32."),
        Q("32، k=3، p=1، s=2 →", ["32", "16", "15"], 1, ""),
        Q("أثقل طبقة معلماتيًا في CNN صغيرة عادةً:", ["Conv الأولى", "Dense بعد Flatten", "MaxPooling"], 1, ""),
        Q("مشاركة الأوزان تعني…", ["نفس النواة في كل موضع", "أوزان Dense مشتركة", "لا انحياز"], 0, ""),
    ])
    takeaway("[Conv+ReLU+Pool]×n → Flatten/GAP → Dense. الأشكال بالصيغة، المعلمات: Conv k²·C_in·C_out+C_out، Dense in·out+out. الأثقل Dense بعد Flatten. طابق summary دائمًا.")
    lesson_footer(LESSON, ["جدول الطبقات.", "بنية نموذجية بتتبع كامل.", "الحاسبة والاختبار البعدي."])
