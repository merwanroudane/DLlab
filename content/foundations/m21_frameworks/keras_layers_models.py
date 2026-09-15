import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table, good_vs_bad
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.frameworks.keras.layers_models",
    title_ar="الطبقات والنماذج: Sequential وFunctional (وSubclassing كتعميق)",
    title_en="Layers & Models: Sequential, Functional (and Subclassing as a Deep Dive)",
    module="foundations.frameworks",
    parent="foundations.frameworks.keras",
    order=4,
    prerequisites=["foundations.frameworks.keras", "foundations.architecture.layers"],
    objectives_ar=["فهم الطبقة ككائن بمعلمات وforward، والنموذج كتركيب طبقات.", "بناء نفس الشبكة بـ Sequential وبـ Functional API ومعرفة متى تحتاج كل واحدة.", "معرفة Subclassing كخيار متقدم دون الحاجة إليه في المسار الأساسي."],
    terms=["model", "parameter", "shape", "batch_dimension"],
    difficulty="beginner",
    summary_ar="الطبقة = تحويل بمعلمات. Sequential لسلسلة خطية من الطبقات؛ Functional لأي رسم (مدخلات/مخرجات متعددة، تفرّعات). Subclassing عندما يحتاج forward منطقًا خاصًا.",
)

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, keras
from keras import layers
keras.utils.set_random_seed(0)

# (أ) Sequential: سلسلة خطية من الطبقات
seq = keras.Sequential([
    layers.Input(shape=(4,)),                 # ملاحظة واحدة = 4 خصائص؛ بُعد الدفعة ضمني
    layers.Dense(8, activation="relu"),        # (None, 4) -> (None, 8): 4*8 + 8 = 40 معلمة
    layers.Dense(3, activation="softmax"),     # (None, 8) -> (None, 3): 8*3 + 3 = 27 معلمة
], name="seq_mlp")

# (ب) Functional: نفس الشبكة كرسم من الاستدعاءات
inp = keras.Input(shape=(4,))
h = layers.Dense(8, activation="relu")(inp)   # الطبقة كائن قابل للاستدعاء على موتر رمزي
out = layers.Dense(3, activation="softmax")(h)
fun = keras.Model(inputs=inp, outputs=out, name="functional_mlp")

# (ج) Functional يسمح بما لا يسمح به Sequential: تفرّع + دمج
inp2 = keras.Input(shape=(4,), name="x")
a = layers.Dense(8, activation="relu", name="branch_a")(inp2)
b = layers.Dense(8, activation="tanh", name="branch_b")(inp2)
merged = layers.Concatenate(name="concat")([a, b])                      # (None, 16)
out2 = layers.Dense(3, activation="softmax", name="head")(merged)
branched = keras.Model(inp2, out2, name="two_branch")

x = np.random.default_rng(0).normal(size=(5, 4)).astype("float32")     # دفعة من 5 ملاحظات
for m in (seq, fun, branched):
    y = m(x)                                                            # استدعاء النموذج = التمرير الأمامي
    print(f"{m.name:<15} params={m.count_params():<4} output shape={tuple(y.shape)}  rows sum to {np.round(np.asarray(y).sum(1), 3)}")
print("layer objects  :", [l.name for l in seq.layers])
print("Dense kernel   :", seq.layers[0].kernel.shape, "bias:", seq.layers[0].bias.shape)'''

SUBCLASS = '''class MyMLP(keras.Model):
    def __init__(self, hidden=8, classes=3):
        super().__init__()
        self.h = layers.Dense(hidden, activation="relu")
        self.out = layers.Dense(classes, activation="softmax")

    def call(self, x, training=False):        # forward بمنطق حر: شروط، حلقات، طبقات مشتركة
        return self.out(self.h(x))

model = MyMLP(); model(x)                      # الأوزان تُنشأ عند أول استدعاء'''


def render() -> None:
    lesson_header(LESSON)
    h2("الطبقة", "The layer")
    definition("**الطبقة** `Layer` في Keras: كائن يملك (1) معلمات (`kernel`, `bias`…) تُنشأ عند معرفة شكل المدخل، و(2) دالة تحويل (التمرير الأمامي) تحوّل موترًا بشكل `(batch, …)` إلى موتر آخر. الطبقات بلا معلمات موجودة أيضًا (`Dropout`, `Flatten`, `Activation`).")
    compare_table(["الطبقة", "التحويل", "المعلمات", "بنيتها في"],
                  [("`Dense(units)`", "z = xW + b ثم تنشيط اختياري", "in×units + units", "وحدة 10–12"), ("`Activation` / `activation=`", "دالة عنصرية", "0", "وحدة 11"), ("`Dropout(p)`", "إسقاط عشوائي في التدريب فقط", "0", "وحدة 20"),
                   ("`BatchNormalization()`", "توحيد لكل خاصية على الدفعة", "4 لكل خاصية (2 متعلَّمة)", "وحدة 20"), ("`Flatten()`", "(b, h, w, c) → (b, h·w·c)", "0", "وحدة 4"), ("`Input(shape)`", "يصف شكل الملاحظة الواحدة فقط", "0", "—")],
                  ["code", "rtl", "rtl", "rtl"])
    intuition("`Input(shape=(4,))` لا يعني «4 ملاحظات» بل «كل ملاحظة 4 خصائص». بُعد الدفعة يُترك `None` لأن النموذج يجب أن يقبل أي عدد من الملاحظات — وهذا ما تراه في `summary()` كـ `(None, 4)`.")
    h2("النموذج: Sequential مقابل Functional", "Model: Sequential vs Functional")
    code_lab(CodeLab(
        key="keras_models", title_ar="نفس الشبكة بثلاث طرق + شبكة متفرّعة", code=CODE, level="B",
        before=Before(goal_ar="بناء MLP بـ Sequential ثم بـ Functional، ثم شبكة متفرّعة لا يمكن التعبير عنها بـ Sequential، والتحقق من عدد المعلمات وشكل المخرج.", stage_ar="Keras: الطبقات → النموذج.",
                      inputs_ar="دفعة x بشكل (5, 4).", expected_ar="seq وfun بـ 67 معلمة ومخرج (5, 3) بصفوف مجموعها 1؛ two_branch بمعلمات أكثر ومخرج (5, 3) أيضًا.",
                      objects_ar="`keras.Sequential`, `keras.Input`, `keras.Model`, `layers.Concatenate`."),
        explain=[("7-11", "Sequential: قائمة طبقات؛ مخرج كل طبقة مدخل التي بعدها. حسبة المعلمات كما في الوحدة 10."), ("14-17", "Functional: `keras.Input` ينشئ موترًا رمزيًا؛ استدعاء الطبقة عليه يبني عقدة في الرسم؛ `keras.Model(inputs, outputs)` يجمع الرسم بين النقطتين."),
                 ("20-25", "تفرّع: نفس المدخل يمر بطبقتين مختلفتين ثم يُدمجان (`Concatenate` → 16 خاصية). هذا مستحيل في Sequential لأن الرسم ليس سلسلة."), ("27-30", "استدعاء النموذج على دفعة = التمرير الأمامي. صفوف softmax تجمع إلى 1 (وحدة 12).")],
        run=run_printed(CODE),
        after_ar="- 67 = 40 + 27: تحقق يدويًا.\n- `two_branch`: 2×(4·8+8) + (16·3+3) = 80 + 51 = 131.\n- `kernel` هو W بشكل (in, units) — لاحظ الترتيب (in, out) في Keras.",
    ))
    compare_table(["الطريقة", "متى", "القيود", "الأمثلة"],
                  [("Sequential", "سلسلة خطية: كل طبقة مدخل واحد ومخرج واحد", "لا تفرّع، لا مدخلات/مخرجات متعددة", "MLP، CNN بسيطة"),
                   ("Functional", "أي رسم موجه: تفرّعات، دمج، مدخلات/مخرجات متعددة، طبقات مشتركة", "الرسم ثابت عند البناء", "ResNet، نماذج متعددة المدخلات"),
                   ("Subclassing", "منطق forward يحتاج بايثون حرًا (شروط، حلقات ديناميكية)", "أقل قابلية للفحص (`summary` أقل تفصيلًا) والحفظ يحتاج عناية", "أبحاث، بنى غير قياسية")],
                  ["ltr", "rtl", "rtl", "rtl"])
    research_note("**تعميق — Subclassing**: ترث `keras.Model`، تنشئ الطبقات في `__init__` وتكتب `call`. تحتاجه فقط عندما لا يكفي الرسم الثابت. في المسار الأساسي للمقرر لن تحتاجه؛ يظهر ثانيةً عند PyTorch حيث هذا هو **الأسلوب الافتراضي** (`nn.Module` مع `forward`).")
    st.code(SUBCLASS, language="python")
    h3("خطأ شائع: نسيان الشكل", "Common error: forgetting the input shape")
    good_vs_bad("Input أولًا", "أضف `layers.Input(shape=(4,))` أولًا (أو مرّر `input_shape=` لأول طبقة): تُنشأ الأوزان فورًا ويعمل `summary()` قبل التدريب.",
                "بلا Input", "بلا شكل مدخل تبقى الأوزان **غير مبنية**: `model.summary()` يطبع `?` في Output Shape و`0 (unbuilt)` في Param #، ولا تُنشأ الأوزان حتى يُستدعى النموذج على بيانات أو `model.build((None, 4))`.",
                good_code="model = keras.Sequential([layers.Input(shape=(4,)), layers.Dense(8)])\nmodel.summary()   # يعمل",
                bad_code="model = keras.Sequential([layers.Dense(8)])\nmodel.summary()   # Output Shape: ?   Param #: 0 (unbuilt)",
                verdict_ar="الأوزان تحتاج شكل المدخل لتعرف حجمها: Dense(8) على 4 خصائص = W بشكل (4, 8)؛ على 100 خاصية = (100, 8).")
    common_mistake("كتابة `Input(shape=(32, 4))` بقصد «دفعة 32». هذا يعني ملاحظة **ثنائية البعد** (32 × 4) — سيقبلها Dense لكن بمعنى مختلف تمامًا. الشكل في `Input` هو شكل الملاحظة الواحدة دائمًا.")
    quiz("keras.models", [
        Q("`Input(shape=(4,))` يعني…", ["4 ملاحظات", "كل ملاحظة 4 خصائص؛ الدفعة ضمنية", "4 طبقات"], 1, "شكل ملاحظة واحدة."),
        Q("شبكة بفرعين يُدمجان تحتاج…", ["Sequential", "Functional (أو Subclassing)", "لا يمكن في Keras"], 1, "رسم غير خطي."),
        Q("`Dense(8)` على مدخل بـ 4 خصائص: معلماتها", ["8", "32", "40"], 2, "4×8 + 8."),
        Q("Subclassing في المسار الأساسي…", ["إلزامي", "تعميق اختياري؛ الافتراضي في PyTorch", "غير موجود"], 1, "nn.Module لاحقًا."),
    ])
    takeaway("طبقة = معلمات + تحويل. Sequential للسلاسل، Functional للرسوم، Subclassing للمنطق الحر. Input يصف الملاحظة الواحدة والدفعة None.")
    lesson_footer(LESSON, ["الطبقة ككائن.", "ثلاث طرق لبناء النموذج.", "شكل المدخل والأوزان غير المبنية."])
