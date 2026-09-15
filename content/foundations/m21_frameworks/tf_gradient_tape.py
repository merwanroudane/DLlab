import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation, worked_steps
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline

LESSON = Lesson(
    id="foundations.frameworks.tensorflow.gradient_tape",
    title_ar="GradientTape: الاشتقاق التلقائي من دالة بسيطة إلى شبكة",
    title_en="GradientTape: Automatic Differentiation from a Function to a Network",
    module="foundations.frameworks",
    parent="foundations.frameworks.tensorflow",
    order=15,
    prerequisites=["foundations.frameworks.tensorflow.tensors", "foundations.calculus.chain_rule", "foundations.backprop.backpropagation.implementation"],
    objectives_ar=["الخطوات الخمس: مخرج ← خسارة ← تسجيل ← تدرج ← مقارنة باليد.", "فهم ما يُسجَّل (العمليات على المتغيرات المراقَبة) وما لا يُسجَّل، وما يحدث خارج with.", "ربط Tape بالانتشار الخلفي في الشبكات: تدرج الخسارة بالنسبة لكل وزن."],
    terms=["derivative", "chain_rule", "gradient", "backpropagation"],
    labs=["labs.gradient_tape_lab"],
    difficulty="intermediate",
    summary_ar="داخل with tf.GradientTape(): كل عملية على tf.Variable تُسجَّل. tape.gradient(loss, vars) يطبق قاعدة السلسلة إلى الخلف ويعيد تدرجًا بنفس شكل كل متغير. هذا ما يفعله fit لكل دفعة.",
)

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf

# المرحلة 1: دالة بسيطة  L(w) = (w·x − y)²   عند x=2, y=1, w=3
x, y = tf.constant(2.0), tf.constant(1.0)
w = tf.Variable(3.0)
with tf.GradientTape() as tape:               # 3) بدء التسجيل
    y_hat = w * x                             # 1) المخرج          = 6
    loss = (y_hat - y) ** 2                   # 2) الخسارة         = 25
g = tape.gradient(loss, w)                    # 4) التدرج
print("y_hat =", float(y_hat), "| loss =", float(loss), "| dL/dw =", float(g), "| hand: 2(wx−y)·x = 2·5·2 =", 2 * (3 * 2 - 1) * 2)   # 5) مقارنة

# ما يُسجَّل وما لا يُسجَّل
c = tf.constant(3.0)
with tf.GradientTape() as tape:
    l1 = c * x                                # ثابت: لا يُراقب تلقائيًا
print("gradient w.r.t. a constant:", tape.gradient(l1, c))              # None
with tf.GradientTape() as tape:
    tape.watch(c)                             # مراقبة صريحة
    l1 = c * x
print("after tape.watch(c):", float(tape.gradient(l1, c)))
with tf.GradientTape(persistent=True) as tape:
    l2 = w * x                                # مسجَّل
l3 = l2 * 10.0                                # خارج with: غير مسجَّل
print("dl2/dw (inside) =", float(tape.gradient(l2, w)), "| dl3/dw (outside) =", tape.gradient(l3, w), "<- the ×10 was never recorded")
with tf.GradientTape() as tape:
    l4 = w * x
tape.gradient(l4, w)
try:
    tape.gradient(l4, w)                      # الشريط غير الدائم يُستهلك بعد أول gradient
except RuntimeError as e:
    print("second call ->", "RuntimeError:", str(e)[:70], "... (use persistent=True)")

# المرحلة 2: شبكة صغيرة — Tape = الانتشار الخلفي للوحدة 14
rng = np.random.default_rng(0)
X = tf.constant(rng.normal(size=(8, 3)).astype("float32")); Y = tf.constant(rng.normal(size=(8, 1)).astype("float32"))
W1 = tf.Variable(rng.normal(0, 0.5, (3, 4)).astype("float32")); b1 = tf.Variable(tf.zeros(4))
W2 = tf.Variable(rng.normal(0, 0.5, (4, 1)).astype("float32")); b2 = tf.Variable(tf.zeros(1))
with tf.GradientTape() as tape:
    A1 = tf.nn.tanh(X @ W1 + b1)
    Y_hat = A1 @ W2 + b2
    mse = tf.reduce_mean((Y_hat - Y) ** 2)
gW1, gb1, gW2, gb2 = tape.gradient(mse, [W1, b1, W2, b2])
# يدويًا بقواعد الوحدة 14: δ2 = 2(Ŷ−Y)/n ; gW2 = A1ᵀδ2 ; δ1 = (δ2 W2ᵀ) ⊙ (1−A1²) ; gW1 = Xᵀδ1
d2 = 2 * (Y_hat - Y) / 8; gW2_hand = tf.transpose(A1) @ d2; d1 = (d2 @ tf.transpose(W2)) * (1 - A1 ** 2); gW1_hand = tf.transpose(X) @ d1
print("shapes: gW1", tuple(gW1.shape), "gb1", tuple(gb1.shape), "gW2", tuple(gW2.shape), "gb2", tuple(gb2.shape), "(= parameter shapes)")
print("max |tape − hand| for W2:", float(tf.reduce_max(tf.abs(gW2 - gW2_hand))), "| for W1:", float(tf.reduce_max(tf.abs(gW1 - gW1_hand))))
# خطوة تحديث كما يفعل المحسّن
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
before = float(mse)
opt.apply_gradients(zip([gW1, gb1, gW2, gb2], [W1, b1, W2, b2]))
after = float(tf.reduce_mean((tf.nn.tanh(X @ W1 + b1) @ W2 + b2 - Y) ** 2))
print(f"mse before {before:.4f} -> after one step {after:.4f}")'''


def render() -> None:
    lesson_header(LESSON)
    definition("**`tf.GradientTape`**: سياق يسجّل كل عملية TensorFlow تُجرى على المتغيرات المراقَبة (كل `tf.Variable` تلقائيًا؛ الثوابت بـ `tape.watch`). بعد الخروج من `with`، `tape.gradient(target, sources)` يطبق **قاعدة السلسلة إلى الخلف** على العمليات المسجلة ويعيد تدرج الهدف بالنسبة لكل مصدر، بنفس شكل المصدر.")
    pipeline(["1. compute output", "2. compute loss", "3. (inside tape) ops recorded", "4. tape.gradient", "5. compare with hand"], active=2)
    why("في الوحدة 14 كتبت `backward` لطبقتين كثيفتين. أضف طبقة ثالثة أو تنشيطًا آخر أو خسارة مختلفة وستعيد الاشتقاق. الشريط يفعل ذلك لأي تركيب من العمليات — بدقة رمزية لا عددية — وهو ما تستدعيه Keras داخل `fit` لكل دفعة.")
    equation(r"\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat y}\cdot\frac{\partial \hat y}{\partial w} = 2(\hat y - y)\cdot x",
             [(r"\hat y = wx", "العملية الأولى المسجلة: مشتقتها المحلية بالنسبة لـ w هي x."), (r"L = (\hat y - y)^2", "العملية الثانية: مشتقتها 2(ŷ−y)."), (r"\partial L/\partial w", "الشريط يضرب المشتقتين المحليتين من الخلف إلى الأمام.")],
             meaning_ar="الشريط يخزّن لكل عملية (المدخلات، المخرج، قاعدة المشتقة المحلية). gradient يمشي على القائمة عكسيًا ويضرب.",
             example_ar="x=2, y=1, w=3: ŷ=6، L=25، dL/dw = 2·5·2 = 20.", dl_link_ar="نفس الفكرة لـ `loss.backward()` في PyTorch.", title_ar="ما يفعله gradient")
    code_lab(CodeLab(
        key="tf_tape", title_ar="الخطوات الخمس، قواعد التسجيل، ثم شبكة كاملة مقارنةً باليد", code=CODE, level="B",
        before=Before(goal_ar="حساب تدرج بالشريط ومقارنته باليد لدالة بسيطة؛ اكتشاف ما يُسجَّل وما لا؛ ثم تدرج خسارة شبكة صغيرة بالنسبة لأربع معلمات ومقارنته بقواعد الوحدة 14 وتطبيق خطوة محسّن.", stage_ar="TensorFlow: الاشتقاق التلقائي.",
                      inputs_ar="أعداد صغيرة ثم شبكة 3→4→1 على 8 ملاحظات.", expected_ar="dL/dw = 20 مطابق لليد؛ None للثابت غير المراقَب؛ خطأ عند استدعاء gradient مرتين؛ فرق ≈ 1e-7 بين الشريط واليد للشبكة؛ mse ينخفض بعد الخطوة.",
                      objects_ar="`tf.GradientTape`, `tape.watch`, `tape.gradient`, `optimizer.apply_gradients`."),
        explain=[("4-11", "الخطوات الخمس على دالة بسيطة. لاحظ: `gradient` يُستدعى **خارج** with."), ("14-17", "الثابت لا يُراقب → `None`. هذا سبب شائع لـ «التدرج None»: المعلمة ليست Variable."),
                 ("18-21", "`tape.watch` يجعل الثابت مراقَبًا."), ("22-25", "ما يُحسب خارج with لا يُسجَّل: l3 لا يرتبط بالشريط فتدرجه None — حتى لو اشتُق من موتر مسجَّل."), ("26-32", "الشريط غير الدائم يُستهلك بعد أول gradient؛ الثاني يرفع خطأ. `persistent=True` لعدة تدرجات."),
                 ("35-43", "شبكة صغيرة: تدرج بالنسبة لقائمة متغيرات يعيد قائمة بنفس الأشكال."), ("44-47", "القواعد اليدوية للوحدة 14 تعطي نفس الأرقام حتى 1e-7."), ("49-53", "`apply_gradients(zip(grads, vars))` = θ ← θ − η·g لكل معلمة: خطوة fit الواحدة.")],
        run=run_printed(CODE),
        after_ar="- الفرق 1e-7 هو تقريب float32 لا فرق في الطريقة: الشريط **هو** الانتشار الخلفي.\n- الأسطر 39–52 هي حلقة تدريب TensorFlow مخصصة كاملة لدفعة واحدة؛ لفّها في حلقتي حقب ودفعات وستحصل على fit يدويًا (الصفحة الرابعة).\n- `None` في التدرج = المتغير لم يشارك في حساب الهدف داخل الشريط (أو ليس Variable).",
    ))
    st.button("افتح معمل GradientTape", icon=":material/science:", type="primary", on_click=go, args=("labs.gradient_tape_lab",), key="tf_tape_lab")
    worked_steps([("داخل with: ŷ = w·x", "يسجّل: عملية ضرب، مدخلاها (w, x)، قاعدة: ∂/∂w = x."), ("داخل with: L = (ŷ − y)²", "يسجّل: طرح ثم تربيع، قاعدة: ∂L/∂ŷ = 2(ŷ−y)."),
                  ("tape.gradient(L, w)", "يبدأ من L بمشتقة 1، يمر بالتربيع: 1·2(ŷ−y) = 10، ثم بالضرب: 10·x = 20."), ("النتيجة", "20 بنفس شكل w (عدد). لمصفوفة W تكون النتيجة مصفوفة بشكلها.")], title_ar="ما يحدث داخل الشريط")
    h2("الربط بالشبكات", "Link to neural networks")
    intuition("في شبكة، «الهدف» هو الخسارة العددية و«المصادر» هي كل الأوزان. الشريط لا يهتم بعدد الطبقات: كل طبقة عمليتان (matmul وتنشيط) بمشتقتين محليتين. δ في الوحدة 14 هو ما يتراكم في الشريط أثناء المشي إلى الخلف. `model.trainable_variables` في Keras هي قائمة المصادر.")
    debugging_note("`tape.gradient` يعيد `None` لبعض المتغيرات؟ الأسباب: (1) ليس `tf.Variable` ولم تُستدعَ `watch`، (2) لم يُستخدم داخل `with`، (3) قُطع التدفق بـ `.numpy()` أو `tf.stop_gradient` أو عملية عددية على أعداد صحيحة (لا مشتقة للأعداد الصحيحة!).")
    common_mistake("تحويل الموتر إلى NumPy داخل الشريط (`loss = np.mean(...)`) يقطع التسجيل: NumPy لا يعرف الشريط. ابقَ داخل عمليات `tf.*` حتى نهاية الخسارة.")
    quiz("tf.tape", [
        Q("أين يُستدعى `tape.gradient`؟", ["داخل with", "خارج with بعده", "قبل with"], 1, "بعد التسجيل."),
        Q("`tape.gradient(L, c)` لثابت غير مراقَب يعيد", ["0", "None", "خطأ"], 1, "لم يُسجَّل."),
        Q("تدرج خسارة بالنسبة لـ W بشكل (3, 4) له شكل", ["()", "(4, 3)", "(3, 4)"], 2, "نفس شكل المصدر."),
        Q("الشريط يحسب التدرج بـ", ["فروق منتهية", "قاعدة السلسلة على العمليات المسجلة", "التدريب"], 1, "رمزي."),
    ])
    takeaway("with GradientTape: سجّل ← gradient(target, sources): قاعدة السلسلة إلى الخلف ← تدرج بشكل المصدر. Variables تُراقب تلقائيًا؛ watch للثوابت؛ None = خارج التسجيل. هذا هو الانتشار الخلفي حيًّا وما يفعله fit لكل دفعة.")
    lesson_footer(LESSON, ["الخطوات الخمس.", "قواعد التسجيل الأربع.", "الشريط = الوحدة 14 لأي شبكة."])
