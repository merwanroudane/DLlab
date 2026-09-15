import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.diagram import diagram, svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="course.w04.forward_flow",
    title_ar="التدفق الأمامي في MLP: الطبقات والأوزان والانحيازات والأشكال — بصريًا وبالكود",
    title_en="Feedforward Flow in an MLP: Layers, Weights, Biases & Shapes — Visually and in Code",
    module="course.w04",
    order=2,
    prerequisites=["course.w04.overview", "foundations.forward.forward_pass_code", "foundations.architecture.parameter_count"],
    objectives_ar=["تعريف الشبكة الأمامية وMLP وطبقاته الثلاثة.", "تتبع دفعة عبر ثلاث طبقات بالأشكال والمعادلات في تحريك.", "التحقق من التدفق بالكود في NumPy ثم في Keras بنفس الأوزان."],
    terms=["weight", "bias", "matrix_multiplication", "shape", "batch_dimension"],
    labs=["labs.network_builder", "labs.parameter_counter"],
    difficulty="beginner",
    summary_ar="MLP: مدخل → طبقات مخفية (xW+b ثم f) → مخرج. الدفعة (B, d) تبقى (B, ·) عبر كل طبقة؛ آخر بُعد = وحدات الطبقة. المعلمات = Σ(in×out + out).",
)

CODE = '''import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, keras
from keras import layers
rng = np.random.default_rng(0)
X = rng.normal(size=(4, 3)).astype("float32")                      # دفعة من 4 ملاحظات × 3 خصائص
W1 = rng.normal(0, 0.5, (3, 5)).astype("float32"); b1 = np.zeros(5, "float32")
W2 = rng.normal(0, 0.5, (5, 2)).astype("float32"); b2 = np.zeros(2, "float32")

# --- التمرير الأمامي في NumPy (الأسس 12) ---
z1 = X @ W1 + b1; a1 = np.maximum(0, z1)                            # (4,3)@(3,5)+(5,) -> (4,5)
z2 = a1 @ W2 + b2; p = np.exp(z2) / np.exp(z2).sum(1, keepdims=True)  # (4,5)@(5,2)+(2,) -> (4,2), softmax
print("shapes: X", X.shape, "-> z1/a1", a1.shape, "-> z2/p", p.shape, "| params:", W1.size + b1.size + W2.size + b2.size)

# --- نفس الأوزان في Keras: يجب أن يعطي نفس الأرقام ---
model = keras.Sequential([layers.Input(shape=(3,)), layers.Dense(5, activation="relu"), layers.Dense(2, activation="softmax")])
model.layers[0].set_weights([W1, b1]); model.layers[1].set_weights([W2, b2])
p_keras = model.predict(X, verbose=0)
print("Keras == NumPy?", np.allclose(p, p_keras, atol=1e-6), "| row 0:", p[0].round(4), "vs", p_keras[0].round(4))
print("params (Keras):", model.count_params(), "= 3*5+5 + 5*2+2")
# طبقة مخفية بلا تنشيط تُطوى: (XW1+b1)W2+b2 = X(W1W2) + (b1W2+b2)
lin = keras.Sequential([layers.Input(shape=(3,)), layers.Dense(5), layers.Dense(2)])
lin.layers[0].set_weights([W1, b1]); lin.layers[1].set_weights([W2, b2])
single = keras.Sequential([layers.Input(shape=(3,)), layers.Dense(2)]); single.layers[0].set_weights([W1 @ W2, b1 @ W2 + b2])
print("two linear layers == one linear layer?", np.allclose(lin.predict(X, verbose=0), single.predict(X, verbose=0), atol=1e-5), "<- why we need activations")'''


def _mlp_svg(stage: int) -> str:
    cols = [("Input\nx (B, 3)", 3, "#E6F1FB"), ("Hidden\na¹ (B, 5)", 5, "#E3F3F0"), ("Output\np (B, 2)", 2, "#FBE6E2")]
    s = '<svg viewBox="0 0 640 260" width="100%" style="max-width:640px">' + svg_defs()
    xs = [80, 320, 560]
    for ci, (lbl, n, fill) in enumerate(cols):
        x = xs[ci]
        active = (ci == stage)
        for i in range(n):
            y = 130 + (i - (n - 1) / 2) * 34
            s += f'<circle cx="{x}" cy="{y}" r="13" fill="{fill}" stroke="{"#1F7A78" if active else "#B9B2A6"}" stroke-width="{2.5 if active else 1.5}"/>'
        a, b = lbl.split("\n")
        s += svg_text(x, 25, a, size=13, bold=True) + svg_text(x, 245, b, size=11, mono=True, color="#6B675F")
    for ci in range(2):
        n1, n2 = cols[ci][1], cols[ci + 1][1]
        color = "#C8473A" if stage == ci + 1 else "#D9D3C7"
        for i in range(n1):
            for j in range(n2):
                y1 = 130 + (i - (n1 - 1) / 2) * 34; y2 = 130 + (j - (n2 - 1) / 2) * 34
                s += f'<line x1="{xs[ci] + 13}" y1="{y1}" x2="{xs[ci + 1] - 13}" y2="{y2}" stroke="{color}" stroke-width="1"/>'
        s += svg_text((xs[ci] + xs[ci + 1]) / 2, 40, f"W{ci + 1} ({cols[ci][1]}, {cols[ci + 1][1]}) + b{ci + 1} ({cols[ci + 1][1]},)", size=11, mono=True, color="#C8473A" if stage == ci + 1 else "#6B675F")
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    definition("**الشبكة الأمامية** `Feedforward network`: المعلومات تتدفق في اتجاه واحد من المدخل إلى المخرج بلا حلقات. **MLP** (Multi-Layer Perceptron): شبكة أمامية من طبقات كثيفة: طبقة مدخل (الخصائص)، طبقة أو أكثر مخفية (كل واحدة `a = f(xW + b)`)، وطبقة مخرج بتنشيط مناسب للمهمة. **الأوزان** W تربط كل وحدة بكل وحدة في الطبقة التالية، و**الانحياز** b لكل وحدة.")
    why("CNN (الأسبوع 08) وRNN (الأسبوع 10) ليسا شيئًا آخر: هما MLP بقيود على W (مشاركة الأوزان، محلية، تكرار عبر الزمن). من يتقن التدفق والأشكال هنا يقرأ أي بنية لاحقًا كـ«MLP بشروط».")
    h2("التدفق المرئي", "Visual flow")
    frames = [Frame(_mlp_svg(0), caption("**الطبقة 0 — المدخل**: دفعة `x` بشكل `(B, 3)`: B ملاحظات، 3 خصائص. لا معلمات هنا."), action="Input", equation="x: (B, 3)"),
              Frame(_mlp_svg(1), caption("**الطبقة 1 — مخفية**: `z¹ = xW¹ + b¹` → `(B, 3)@(3, 5)+(5,) = (B, 5)` ثم `a¹ = ReLU(z¹)`. المعلمات: 3×5+5 = **20**."), action="Hidden", equation="a¹ = ReLU(xW¹ + b¹): (B, 5)", values=[("params", "", "20")]),
              Frame(_mlp_svg(2), caption("**الطبقة 2 — المخرج**: `z² = a¹W² + b²` → `(B, 5)@(5, 2)+(2,) = (B, 2)` ثم softmax → احتمالات لفئتين. المعلمات: 5×2+2 = **12**. المجموع 32."), action="Output", equation="p = softmax(a¹W² + b²): (B, 2)", values=[("params", "20", "32")])]
    animation_player("w04_flow", frames, title_ar="دفعة تعبر MLP: 3 → 5 → 2", stages=["Input", "Hidden", "Output"], interval_ms=1800)
    equation(r"a^{(\ell)} = f\big(a^{(\ell-1)} W^{(\ell)} + b^{(\ell)}\big), \qquad a^{(0)} = x, \qquad W^{(\ell)} \in \mathbb{R}^{n_{\ell-1}\times n_\ell}",
             [(r"a^{(\ell-1)}", "مخرج الطبقة السابقة بشكل (B, n_{ℓ−1})."), (r"W^{(\ell)}", "أوزان الطبقة: (وحدات سابقة × وحدات حالية)."), (r"b^{(\ell)}", "انحياز لكل وحدة، يُبث على الدفعة."), ("f", "تنشيط غير خطي (ReLU في المخفية؛ sigmoid/softmax/بلا في المخرج).")],
             meaning_ar="معادلة واحدة تتكرر لكل طبقة. بُعد الدفعة B لا يمسّه شيء؛ آخر بُعد يتغير إلى عدد وحدات الطبقة.",
             example_ar="(B,3) → (B,5) → (B,2): المعلمات 20 + 12 = 32.", dl_link_ar="`Dense(5, activation='relu')` هي هذه المعادلة بالضبط مع W بشكل (in, 5).", title_ar="معادلة الطبقة")
    code_lab(CodeLab(
        key="w04_flow", title_ar="التمرير الأمامي في NumPy ثم في Keras بنفس الأوزان — ولماذا نحتاج التنشيط", code=CODE, level="B",
        before=Before(goal_ar="حساب التدفق يدويًا بمصفوفات صغيرة، ثم زرع نفس الأوزان في Keras والتحقق من تطابق الأرقام، ثم إثبات أن طبقتين خطيتين = طبقة واحدة.", stage_ar="الأسبوع 04: التدفق الأمامي.",
                      inputs_ar="دفعة (4, 3) وأوزان (3,5) و(5,2).", expected_ar="أشكال (4,5) و(4,2)؛ 32 معلمة؛ Keras == NumPy True؛ الطبقتان الخطيتان = طبقة واحدة True."),
        explain=[("4-6", "دفعة وأوزان عشوائية بأشكال (in, out) كما تخزّنها Keras."), ("9-11", "الطبقتان بالمعادلة، مع تعليق الأشكال. softmax على المحور 1 (لكل ملاحظة)."), ("14-18", "`set_weights([W, b])` يزرع أوزاننا؛ `predict` يعطي نفس الأرقام: لا سحر في Dense."), ("20-23", "بلا تنشيط، (xW₁+b₁)W₂+b₂ = x(W₁W₂)+(b₁W₂+b₂): الشبكة «العميقة» الخطية تساوي طبقة خطية واحدة — الأسس 11: لماذا اللاخطية.")],
        run=run_printed(CODE),
        after_ar="- `Keras == NumPy? True`: كل ما تفعله Dense هو `x @ W + b` ثم f.\n- 32 معلمة = ما يطبعه summary.\n- الطي الخطي هو السبب الرياضي لوجود التنشيطات (الأسبوع 06).",
    ))
    with st.container(horizontal=True):
        st.button("معمل بناء الشبكة", icon=":material/science:", on_click=go, args=("labs.network_builder",), key="w04_lab_builder")
        st.button("معمل عدّ المعلمات", icon=":material/science:", on_click=go, args=("labs.parameter_counter",), key="w04_lab_params")
        st.button("المكافئ في PyTorch (nn.Module)", icon=":material/swap_horiz:", on_click=go, args=("foundations.frameworks.pytorch.nn_module",), key="w04_go_pt")
    intuition("فكّر في كل طبقة كـ«إعادة وصف» للدفعة بعدد جديد من الأعمدة: 3 خصائص خام → 5 خصائص متعلَّمة → 2 درجة قرار. الصفوف (الملاحظات) لا تختلط أبدًا في MLP.")
    common_mistake("كتابة W بشكل (out, in) في NumPy ثم `X @ W`: خطأ شكل. Keras (in, out) و`x @ W`؛ PyTorch (out, in) و`x @ W.T`. الأشكال أولًا دائمًا.")
    quiz("w04.flow", [
        Q("دفعة (64, 20) عبر Dense(50) ثم Dense(3): شكل المخرج", ["(64, 3)", "(3,)", "(20, 3)"], 0, ""),
        Q("معلمات الشبكة السابقة:", ["1050 + 153 = 1203", "1000 + 150", "70"], 0, "20×50+50 + 50×3+3."),
        Q("طبقتان خطيتان متتاليتان بلا تنشيط…", ["أقوى من واحدة", "تساويان طبقة خطية واحدة", "لا تُدرَّبان"], 1, ""),
        Q("بُعد الدفعة عبر الطبقات…", ["يتغير كل طبقة", "ثابت", "يختفي"], 1, ""),
    ])
    takeaway("MLP = تكرار a = f(aW + b). الدفعة (B, ·) ثابتة الصفوف؛ الأعمدة = وحدات الطبقة. المعلمات Σ(in×out+out). بلا تنشيط تُطوى الطبقات.")
    lesson_footer(LESSON, ["التعريف والمعادلة.", "التدفق المرئي بالأشكال.", "NumPy = Keras بنفس الأوزان."])
