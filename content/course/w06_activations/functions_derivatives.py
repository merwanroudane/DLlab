import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto
from core.rtl import table

LESSON = Lesson(
    id="course.w06.functions_derivatives",
    title_ar="الدوال ومشتقاتها: Sigmoid، Tanh، ReLU، Leaky ReLU، ELU، SELU، Swish، Softmax — التشبع وReLU الميت",
    title_en="The Functions & Their Derivatives: Saturation and Dead ReLU",
    module="course.w06",
    order=2,
    prerequisites=["course.w06.overview", "foundations.activations.derivatives_saturation_dead", "foundations.backprop.backpropagation.vanishing_exploding"],
    objectives_ar=["رسم الدوال ومشتقاتها جنبًا إلى جنب وقراءة مناطق التشبع.", "قياس تلاشي التدرج عبر 8 طبقات بـ sigmoid مقابل ReLU في Keras (GradientTape).", "إثبات ReLU الميت تجريبيًا وعلاجه بـ Leaky ReLU."],
    terms=["derivative", "gradient", "softmax"],
    labs=["labs.activation_lab"],
    difficulty="intermediate",
    summary_ar="المشتقة تحدد ما يمر من التدرج: sigmoid ≤ 0.25 وتتشبع؛ tanh ≤ 1 وتتشبع؛ ReLU 1 أو 0 (ميت)؛ Leaky/ELU/SELU/Swish تصلح الصفر. Softmax للمخرج المتعدد فقط.",
)

FUNCS = {
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda z: (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z)))),
    "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    "relu": (lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)),
    "leaky_relu (0.1)": (lambda z: np.where(z > 0, z, 0.1 * z), lambda z: np.where(z > 0, 1.0, 0.1)),
    "elu": (lambda z: np.where(z > 0, z, np.exp(np.minimum(z, 0)) - 1), lambda z: np.where(z > 0, 1.0, np.exp(np.minimum(z, 0)))),
    "selu": (lambda z: 1.0507 * np.where(z > 0, z, 1.6733 * (np.exp(np.minimum(z, 0)) - 1)), lambda z: 1.0507 * np.where(z > 0, 1.0, 1.6733 * np.exp(np.minimum(z, 0)))),
    "swish": (lambda z: z / (1 + np.exp(-z)), lambda z: (1 / (1 + np.exp(-z))) * (1 + z * (1 - 1 / (1 + np.exp(-z))))),
}

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf, keras
from keras import layers
X = tf.constant(np.random.default_rng(0).normal(size=(64, 8)).astype("float32")); y = tf.constant(np.random.default_rng(1).uniform(size=(64, 1)).astype("float32"))

def grad_norms(act, depth=8, init="glorot_uniform"):
    keras.utils.set_random_seed(0)
    m = keras.Sequential([layers.Input(shape=(8,))] + [layers.Dense(16, activation=act, kernel_initializer=init) for _ in range(depth)] + [layers.Dense(1, activation="sigmoid")])
    with tf.GradientTape() as tape:
        loss = keras.losses.BinaryCrossentropy()(y, m(X, training=True))
    grads = tape.gradient(loss, m.trainable_variables)
    return [float(tf.norm(g)) for g in grads[::2]]                # معيار تدرج kernel لكل طبقة (الأولى -> الأخيرة)

print("‖∇W‖ per layer, layer 1 (first) ... layer 9 (output):")
for act in ("sigmoid", "tanh", "relu", "elu"):
    g = grad_norms(act)
    print(f"  {act:<8} first {g[0]:.2e}  middle {g[4]:.2e}  last {g[-1]:.2e}   ratio first/last = {g[0] / g[-1]:.4f}")

# ReLU الميت: تهيئة تدفع z إلى السالب -> تدرج 0 دائمًا
print("\\nDead ReLU experiment (bias pushed to -5 in layer 1):")
for act in ("relu", "leaky_relu"):
    keras.utils.set_random_seed(0)
    m = keras.Sequential([layers.Input(shape=(8,)), layers.Dense(16, activation=act), layers.Dense(1, activation="sigmoid")])
    k, b = m.layers[0].get_weights(); m.layers[0].set_weights([k, b - 5.0])
    with tf.GradientTape() as tape:
        loss = keras.losses.BinaryCrossentropy()(y, m(X))
    gk = tape.gradient(loss, m.layers[0].kernel)
    act_out = m.layers[0](X).numpy()
    print(f"  {act:<11} units with zero output on all 64 samples: {(np.abs(act_out).max(0) == 0).sum():>2}/16 | kernel grad norm: {float(tf.norm(gk)):.2e}")'''


def render() -> None:
    lesson_header(LESSON)
    definition("**دالة التنشيط** تُطبَّق عنصريًا على z = xW + b. مشتقتها f′(z) هي **العامل** الذي يُضرب فيه التدرج عند المرور عبر الطبقة إلى الخلف (الأسس 14). **التشبع**: منطقة تكون فيها f′ ≈ 0 فيتلاشى التدرج. **ReLU الميت**: وحدة z < 0 لكل المدخلات → مخرج 0 وتدرج 0 → لا تتعلم أبدًا.")
    h2("الدوال ومشتقاتها", "Functions and derivatives")
    sel = st.multiselect("اختر دوالًا للرسم", list(FUNCS), default=["sigmoid", "tanh", "relu", "elu"], key="w06_sel")
    z = np.linspace(-6, 6, 400)
    c1, c2 = st.columns(2)
    colors = ["#2F6FB5", "#7C5CBF", "#1F7A78", "#C8473A", "#D9A21B", "#6B675F", "#E8710A"]
    for col, idx, title in ((c1, 0, "f(z)"), (c2, 1, "f′(z)")):
        fig = go.Figure()
        for name, color in zip(sel, colors):
            fig.add_trace(go.Scatter(x=z, y=FUNCS[name][idx](z), name=name, line=dict(color=color, width=2.5)))
        fig.update_layout(height=300, title=title, margin=dict(l=10, r=10, t=35, b=10), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
        with col:
            st.plotly_chart(fig, width="stretch", key=f"w06_fig_{idx}")
    table(["الدالة", "المدى", "f′ القصوى", "التشبع", "الصفر الميت", "الاستخدام النموذجي"],
          [("sigmoid", "(0, 1)", "0.25", "الطرفان", "لا", "مخرج ثنائي فقط"), ("tanh", "(−1, 1)", "1", "الطرفان", "لا", "RNN/LSTM داخليًا؛ مخفي قديم"), ("ReLU", "[0, ∞)", "1", "لا (للموجب)", "**نعم** (z<0)", "الافتراضي للمخفي"),
           ("Leaky ReLU", "(−∞, ∞)", "1", "لا", "لا (ميل α)", "عند ظهور وحدات ميتة"), ("ELU", "(−α, ∞)", "1", "سالب ناعم", "لا", "متوسط ≈ 0؛ أبطأ قليلًا"), ("SELU", "مقيَّس", "1.05", "سالب ناعم", "لا", "شبكات كثيفة عميقة بتهيئة lecun_normal"),
           ("Swish", "(−0.28, ∞)", "≈1.1", "لا", "لا", "بديل ReLU في نماذج حديثة"), ("Softmax", "احتمالات تجمع لـ 1", "—", "—", "—", "مخرج متعدد الفئات فقط")],
          ["ltr", "code", "num", "rtl", "rtl", "rtl"])
    intuition("اقرأ رسم المشتقة لا الدالة: عند |z| > 4 مشتقة sigmoid < 0.02؛ ثماني طبقات كهذه تضرب التدرج في 0.02⁸. ReLU مشتقتها 1 للموجب — لا تضعف الإشارة — لكنها 0 للسالب: إما تمر كاملة أو لا تمر.")
    code_lab(CodeLab(
        key="w06_vanish", title_ar="تلاشي التدرج عبر 8 طبقات بأربع دوال، وتجربة ReLU الميت", code=CODE, level="B",
        before=Before(goal_ar="قياس معيار تدرج أوزان كل طبقة في شبكة من 8 طبقات مخفية لأربع دوال تنشيط، ثم إثبات ReLU الميت بتهيئة سيئة ومقارنته بـ Leaky ReLU.", stage_ar="الأسبوع 06: المشتقات والتشبع.",
                      inputs_ar="دفعة (64, 8) عشوائية وهدف عشوائي.", expected_ar="sigmoid: نسبة الأول/الأخير صغيرة جدًا (تلاشٍ)؛ ReLU/ELU: نسبة قرب 1؛ ReLU مع انحياز −5: معظم الوحدات ميتة وتدرج ≈ 0؛ Leaky: تدرج غير صفري."),
        explain=[("6-12", "شبكة بعمق 8 بالدالة المطلوبة؛ تدرج الخسارة بالنسبة لكل وزن بـ GradientTape (الوحدة 21)؛ نأخذ معيار kernel لكل طبقة."), ("14-17", "المقارنة: النسبة الأول/الأخير تقيس كم وصل من التدرج إلى الطبقة الأولى."),
                 ("20-28", "ReLU الميت صناعيًا: انحياز −5 يجعل z سالبًا لكل المدخلات → مخرج 0 لكل الوحدات وتدرج kernel صفر. Leaky ReLU يبقي ميلًا 0.2 (الافتراضي في Keras 3) فيبقى تدرج.")],
        run=run_printed(CODE),
        after_ar="- النسبة لـ sigmoid تُظهر التلاشي رقميًا: الطبقات الأولى تتعلم أبطأ بأضعاف كثيرة.\n- ReLU/ELU تحافظ على المعيار عبر العمق: هذا سبب سيادة ReLU في الشبكات العميقة.\n- الوحدة الميتة لا تعود: تدرجها 0 فلا يعدّل الانحياز أبدًا. الوقاية: η معقول، تهيئة He، أو Leaky/ELU.",
    ))
    st.button("معمل التنشيط (حرّك z وشاهد f وf′)", icon=":material/science:", type="primary", on_click=goto, args=("labs.activation_lab",), key="w06_lab_act")
    why("Softmax ليست تنشيطًا «عنصريًا»: كل مخرج يعتمد على كل z في الطبقة (القسمة على المجموع). لذلك لا تُستخدم في الطبقات المخفية، ولا مع وحدة واحدة (تعطي 1 دائمًا).")
    common_mistake("`Dense(1, activation='softmax')`: مخرج ثابت = 1.0 لكل ملاحظة، والخسارة لا تنخفض. الثنائي = sigmoid بوحدة واحدة أو softmax بوحدتين.")
    quiz("w06.fn", [
        Q("مشتقة sigmoid عند z = 5 ≈", ["0.25", "0.0066", "1"], 1, "تشبع."),
        Q("ReLU الميت:", ["z > 0 دائمًا", "z < 0 لكل المدخلات → تدرج 0", "مشتقة كبيرة"], 1, ""),
        Q("أي دالة تحافظ على معيار التدرج عبر 8 طبقات؟", ["sigmoid", "ReLU / ELU", "softmax"], 1, ""),
        Q("Softmax في طبقة مخفية…", ["ممتازة", "غير مناسبة: ليست عنصرية وتُطبّع الطبقة", "تسرّع"], 1, ""),
    ])
    takeaway("المشتقة هي ما يمر من التدرج. sigmoid/tanh تتشبعان؛ ReLU تمرر 1 أو 0 (ميت)؛ Leaky/ELU/SELU/Swish تصلح الصفر. Softmax للمخرج المتعدد فقط.")
    lesson_footer(LESSON, ["الدوال والمشتقات بالرسم والجدول.", "التلاشي وReLU الميت بالأرقام.", "Softmax استثناء."])
