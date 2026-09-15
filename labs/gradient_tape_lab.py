"""GradientTape Lab — from a scalar function to a neural network: record ops,
compute the gradient, and compare with the hand derivative / finite differences
(spec §21.10)."""

import numpy as np
import streamlit as st

from components.callouts import intuition, practical_note
from core.models import Lab
from core.progress import mark_visited
from core.rtl import esc, table
from labs.fw import tf

LAB = Lab(
    id="labs.gradient_tape_lab",
    title_ar="معمل GradientTape",
    title_en="GradientTape Lab",
    category="frameworks",
    description_ar="اختر دالة، حرّك قيمة المتغير، وشاهد tf.GradientTape يسجّل العمليات ويحسب التدرج — ثم قارنه بالاشتقاق اليدوي وبالفروق المنتهية. المرحلة الثانية: نفس الشيء لخسارة شبكة صغيرة بالنسبة لأوزانها.",
    related_lessons=["foundations.frameworks.tensorflow.gradient_tape", "foundations.calculus.derivative", "foundations.backprop.backpropagation.implementation"],
)

FUNCS = {
    "L = (3w − 1)²": (lambda T, w: (3.0 * w - 1.0) ** 2, lambda w: 6.0 * (3.0 * w - 1.0), "dL/dw = 2(3w−1)·3 = 6(3w−1)"),
    "L = w³ − 2w": (lambda T, w: w ** 3 - 2.0 * w, lambda w: 3.0 * w ** 2 - 2.0, "dL/dw = 3w² − 2"),
    "L = sin(w)·w": (lambda T, w: T.sin(w) * w, lambda w: np.cos(w) * w + np.sin(w), "dL/dw = w·cos(w) + sin(w)"),
    "L = exp(−w²)": (lambda T, w: T.exp(-(w ** 2)), lambda w: -2.0 * w * np.exp(-(w ** 2)), "dL/dw = −2w·exp(−w²)"),
    "L = log(1 + e^w)": (lambda T, w: T.math.log(1.0 + T.exp(w)), lambda w: 1.0 / (1.0 + np.exp(-w)), "dL/dw = σ(w) (softplus → sigmoid)"),
}


def render() -> None:
    mark_visited(LAB.id)
    st.markdown(f"# 🧪 {LAB.title_ar}")
    st.markdown(f"<div class='en' style='color:#6B675F'>{esc(LAB.title_en)}</div>", unsafe_allow_html=True)
    st.markdown(LAB.description_ar)
    T = tf()
    st.markdown("### المرحلة 1: دالة عددية")
    c1, c2 = st.columns(2)
    with c1:
        fname = st.selectbox("الدالة", list(FUNCS), key="gt_f")
    with c2:
        w0 = st.slider("قيمة w", -3.0, 3.0, 2.0, 0.1, key="gt_w")
    f, df_hand, formula = FUNCS[fname]
    w = T.Variable(float(w0))
    with T.GradientTape() as tape:
        L = f(T, w)
    g = tape.gradient(L, w)
    eps = 1e-4
    fd = (float(f(T, T.constant(w0 + eps))) - float(f(T, T.constant(w0 - eps)))) / (2 * eps)
    st.code(f"""w = tf.Variable({w0:.1f})
with tf.GradientTape() as tape:        # 1) ابدأ التسجيل
    L = {fname.split('=')[1].strip()}                  # 2) احسب المخرج/الخسارة (العمليات تُسجَّل)
g = tape.gradient(L, w)               # 3) طبّق قاعدة السلسلة إلى الخلف

L                = {float(L):.6f}
tape.gradient    = {float(g):.6f}
hand derivative  = {float(df_hand(w0)):.6f}      # {formula}
finite difference= {fd:.6f}      # (L(w+ε) − L(w−ε)) / 2ε""", language="text")
    table(["الطريقة", "القيمة", "الملاحظة"], [("GradientTape", f"{float(g):.6f}", "دقيق: قاعدة السلسلة رمزيًا على العمليات المسجلة"), ("يدويًا", f"{float(df_hand(w0)):.6f}", formula), ("فروق منتهية", f"{fd:.6f}", "تقريب عددي (وحدة 5) — أداة تحقق لا أداة تدريب")], ["rtl", "num", "rtl"])
    intuition("الشريط لا «يعرف» الدالة: يعرف فقط سلسلة العمليات الأولية (ضرب، جمع، sin، exp…) ومشتقة كل واحدة. `gradient` يضربها بقاعدة السلسلة إلى الخلف (وحدة 5 و14). لهذا يعمل مع **أي** دالة تكتبها من عمليات TensorFlow.")

    st.markdown("### المرحلة 2: خسارة شبكة صغيرة بالنسبة لأوزانها")
    c3, c4 = st.columns(2)
    with c3:
        hidden = st.select_slider("وحدات مخفية", options=[2, 4, 8], value=4, key="gt_h")
    with c4:
        seed = st.slider("البذرة", 0, 5, 0, key="gt_seed")
    rng = np.random.default_rng(seed)
    X = T.constant(rng.normal(size=(16, 3)).astype("float32")); y = T.constant((rng.uniform(size=(16, 1)) > 0.5).astype("float32"))
    W1 = T.Variable(rng.normal(0, 0.5, (3, hidden)).astype("float32")); b1 = T.Variable(np.zeros((hidden,), "float32"))
    W2 = T.Variable(rng.normal(0, 0.5, (hidden, 1)).astype("float32")); b2 = T.Variable(np.zeros((1,), "float32"))

    def loss_fn():
        h = T.nn.relu(X @ W1 + b1); p = T.nn.sigmoid(h @ W2 + b2)
        return -T.reduce_mean(y * T.math.log(p + 1e-7) + (1 - y) * T.math.log(1 - p + 1e-7))

    with T.GradientTape() as tape:
        loss = loss_fn()
    grads = tape.gradient(loss, [W1, b1, W2, b2])
    # finite-difference check on one entry of each parameter
    def fd_check(var, idx):
        old = float(var.numpy()[idx]); e = 1e-3
        arr = var.numpy(); arr[idx] = old + e; var.assign(arr); lp = float(loss_fn())
        arr[idx] = old - e; var.assign(arr); lm = float(loss_fn())
        arr[idx] = old; var.assign(arr)
        return (lp - lm) / (2 * e)
    rows = []
    for name, var, gr, idx in [("W1", W1, grads[0], (0, 0)), ("b1", b1, grads[1], (0,)), ("W2", W2, grads[2], (0, 0)), ("b2", b2, grads[3], (0,))]:
        rows.append((name, str(tuple(var.shape)), str(tuple(gr.shape)), f"{float(gr.numpy()[idx]):+.6f}", f"{fd_check(var, idx):+.6f}", f"{float(T.norm(gr)):.4f}"))
    st.code(f"""with tf.GradientTape() as tape:
    h = tf.nn.relu(X @ W1 + b1)                 # X (16, 3) -> h (16, {hidden})
    p = tf.nn.sigmoid(h @ W2 + b2)              # -> p (16, 1)
    loss = binary_cross_entropy(y, p)           # عدد واحد: {float(loss):.4f}
grads = tape.gradient(loss, [W1, b1, W2, b2])   # قائمة بنفس أشكال المتغيرات""", language="python")
    table(["المعلمة", "شكلها", "شكل تدرجها", "التدرج [0] (Tape)", "فروق منتهية [0]", "‖∇‖"], rows, ["code", "code", "code", "num", "num", "num"])
    practical_note("هذا بالضبط ما فعلته في الوحدة 14 بـ `backward` يدويًا و`gradient_check`. الفرق: هنا لا تكتب أي مشتقة. Keras `fit` تستدعي `tape.gradient(loss, model.trainable_variables)` لكل دفعة ثم `optimizer.apply_gradients`.")
