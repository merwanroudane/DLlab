import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, definition, interpretation_note, intuition, math_note, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.diagram import diagram, svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation, worked_steps
from components.quiz import Q, quiz
from content.course.w04_ffnn._viz import NET, approx_svg, forward_numbers, matmul_data, matmul_svg, neuron_svg, relu_approx
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="course.w04.forward_flow",
    title_ar="التدفق الأمامي في MLP: الطبقات والأوزان والانحيازات والأشكال — بصريًا وبالكود",
    title_en="Feedforward Flow in an MLP: Layers, Weights, Biases & Shapes — Visually and in Code",
    module="course.w04",
    order=2,
    prerequisites=["course.w04.overview", "foundations.forward.forward_pass_code", "foundations.architecture.parameter_count"],
    objectives_ar=[
        "تعريف الشبكة الأمامية وMLP وطبقاته الثلاثة.",
        "تتبّع عميل واحد عبر شبكة 2 → 3 → 1 عصبونًا عصبونًا بأرقام حقيقية، ورؤية وحدات ReLU «المطفأة».",
        "فهم ضرب المصفوفات كخلايا: كل خلية = صف ملاحظة · عمود وحدة، وتتبّع دفعة عبر ثلاث طبقات بالأشكال.",
        "رؤية لماذا تستطيع طبقة مخفية واحدة تقريب أي منحنى (التقريب الشامل) بإضافة وحدات ReLU واحدة تلو الأخرى.",
        "التحقق من التدفق بالكود في NumPy ثم في Keras بنفس الأوزان.",
    ],
    terms=["weight", "bias", "matrix_multiplication", "shape", "batch_dimension", "neuron", "layer", "dense_layer", "relu", "activation_function"],
    labs=["labs.network_builder", "labs.parameter_counter", "labs.neuron_lab"],
    difficulty="beginner",
    summary_ar="MLP: مدخل → طبقات مخفية (xW+b ثم f) → مخرج. كل خلية في XW = صف · عمود. الدفعة (B, d) تبقى (B, ·)؛ آخر بُعد = وحدات الطبقة. المعلمات = Σ(in×out + out). مجموع ReLU يقرّب أي منحنى.",
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
            s += f'<circle cx="{x}" cy="{y}" r="13" fill="{fill}" stroke="{"#7C3AED" if active else "#B9B2A6"}" stroke-width="{2.5 if active else 1.5}"/>'
        a, b = lbl.split("\n")
        s += svg_text(x, 25, a, size=13, bold=True) + svg_text(x, 245, b, size=11, mono=True, color="#6B675F")
    for ci in range(2):
        n1, n2 = cols[ci][1], cols[ci + 1][1]
        color = "#DB2777" if stage == ci + 1 else "#D9D3C7"
        for i in range(n1):
            for j in range(n2):
                y1 = 130 + (i - (n1 - 1) / 2) * 34; y2 = 130 + (j - (n2 - 1) / 2) * 34
                s += f'<line x1="{xs[ci] + 13}" y1="{y1}" x2="{xs[ci + 1] - 13}" y2="{y2}" stroke="{color}" stroke-width="1"/>'
        s += svg_text((xs[ci] + xs[ci + 1]) / 2, 40, f"W{ci + 1} ({cols[ci][1]}, {cols[ci + 1][1]}) + b{ci + 1} ({cols[ci + 1][1]},)", size=11, mono=True, color="#DB2777" if stage == ci + 1 else "#6B675F")
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    definition("**الشبكة الأمامية** `Feedforward network`: المعلومات تتدفق في اتجاه واحد من المدخل إلى المخرج بلا حلقات. **MLP** (Multi-Layer Perceptron): شبكة أمامية من طبقات كثيفة: طبقة مدخل (الخصائص)، طبقة أو أكثر مخفية (كل واحدة `a = f(xW + b)`)، وطبقة مخرج بتنشيط مناسب للمهمة. **الأوزان** W تربط كل وحدة بكل وحدة في الطبقة التالية، و**الانحياز** b لكل وحدة.")
    why("CNN (الأسبوع 08) وRNN (الأسبوع 10) ليسا شيئًا آخر: هما MLP بقيود على W (مشاركة الأوزان، محلية، تكرار عبر الزمن). من يتقن التدفق والأشكال هنا يقرأ أي بنية لاحقًا كـ«MLP بشروط».")

    # ------------------------------------------------------------------ neuron by neuron
    h2("عميل واحد عبر الشبكة، عصبونًا عصبونًا", "One customer through the network, neuron by neuron")
    f = forward_numbers()
    x, W1, b1, W2, b2 = NET["x"], NET["W1"], NET["b1"], NET["W2"], NET["b2"]
    ncaps = [f"**المدخل**: عميل بخاصيتين محجّمتين: نسبة دين `{x[0]:+.1f}` (فوق المتوسط) ودخل `{x[1]:+.1f}` (تحت المتوسط). الأوزان أدناه أمثلة ثابتة لنتتبع الحساب."]
    for j in range(3):
        on = "تعمل: تمرّر قيمتها" if f["a1"][j] > 0 else "**مطفأة**: z سالب فـ ReLU تعطي صفرًا — هذه الوحدة لا تساهم في قرار هذا العميل"
        ncaps.append(f"**الوحدة المخفية h{j + 1}**: `z = ({x[0]:+.1f})({W1[0, j]:+.1f}) + ({x[1]:+.1f})({W1[1, j]:+.1f}) + ({b1[j]:+.2f}) = {f['z1'][j]:+.2f}`، ثم `a = ReLU(z) = {f['a1'][j]:.2f}`. {on}.")
    ncaps.append(f"**وحدة المخرج**: تجمع مخرجات المخفية بأوزانها: `z = ({f['a1'][0]:.2f})({W2[0, 0]:+.1f}) + ({f['a1'][1]:.2f})({W2[1, 0]:+.1f}) + ({f['a1'][2]:.2f})({W2[2, 0]:+.1f}) + ({b2[0]:+.1f}) = {f['z2']:+.3f}`. الوحدتان المطفأتان لا تؤثران مهما كان وزنهما.")
    ncaps.append(f"**sigmoid**: `p = σ({f['z2']:.3f}) = {f['p']:.3f}` — احتمال التعثر لهذا العميل. عميل آخر بخصائص مختلفة سيشعل وحدات مختلفة: لكل منطقة من فضاء الخصائص «دائرة» مختلفة من الوحدات النشطة.")
    animation_player("w04_neuron", [Frame(neuron_svg(i), caption(c), action=["input", "h1", "h2", "h3", "output z", "σ"][i], highlight=min(i, 2) if i < 4 else 2)
                                    for i, c in enumerate(ncaps)], title_ar="شبكة 2 → 3 → 1 بأرقام", stages=["input", "hidden (ReLU)", "output"], interval_ms=2400)
    worked_steps([
        ("الطبقة المخفية بصيغة المصفوفات (صف × مصفوفة)", rf"z^{{(1)}} = x W^{{(1)}} + b^{{(1)}} = [{f['z1'][0]:.2f},\ {f['z1'][1]:.2f},\ {f['z1'][2]:.2f}]"),
        ("ReLU عنصرًا عنصرًا", rf"a^{{(1)}} = \max(0, z^{{(1)}}) = [{f['a1'][0]:.2f},\ {f['a1'][1]:.2f},\ {f['a1'][2]:.2f}]"),
        ("المخرج", rf"z^{{(2)}} = a^{{(1)}} W^{{(2)}} + b^{{(2)}} = {f['z2']:.3f},\qquad p = \sigma(z^{{(2)}}) = {f['p']:.3f}"),
    ], title_ar="نفس الحساب بالمصفوفات")
    intuition("وحدة ReLU المطفأة ليست «معطّلة» — هي فقط غير معنية بهذه المنطقة من البيانات. الشبكة تقسم فضاء الخصائص إلى مناطق، وفي كل منطقة مجموعة مختلفة من الوحدات تعمل.")

    # ------------------------------------------------------------------ matmul view
    h2("الدفعة كاملة: ضرب المصفوفات خلية خلية", "The whole batch: matrix multiplication cell by cell")
    md = matmul_data()
    cells = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 4), (3, 3)]
    mframes = [Frame(matmul_svg(md, None, None, -1), caption("**4 ملاحظات × 3 خصائص** مضروبة في **أوزان 3 × 5** (خمس وحدات). المصفوفة الناتجة 4 × 5: صف لكل ملاحظة، عمود لكل وحدة. قاعدة الأشكال: `(4, 3) @ (3, 5) → (4, 5)` — البعدان الداخليان يجب أن يتطابقا ويختفيا."), action="shapes")]
    for i, j in cells:
        reveal = i * 5 + j
        mframes.append(Frame(matmul_svg(md, i, j, reveal), caption(f"**الخلية Z[{i},{j}]** = صف الملاحظة {i} (أزرق) · عمود الوحدة {j} (بنفسجي): ضرب عنصرًا بعنصر ثم جمع. هذا هو المجموع الموزون للوحدة {j} على الملاحظة {i} — نفس ما حسبناه عصبونًا عصبونًا أعلاه."),
                             action=f"Z[{i},{j}]"))
    mframes.append(Frame(matmul_svg(md, None, None, 19), caption("**كل الخلايا**: عملية واحدة `X @ W` تحسب 20 مجموعًا موزونًا دفعة واحدة. لهذا تعمل الشبكات على دفعات: عملية مصفوفات واحدة كبيرة أسرع بكثير من حلقات (وهذا ما يسرّعه GPU)."), action="all cells"))
    animation_player("w04_matmul", mframes, title_ar="X @ W: كل خلية = ملاحظة · وحدة", interval_ms=1800)

    h2("التدفق المرئي عبر ثلاث طبقات", "Visual flow")
    frames = [Frame(_mlp_svg(0), caption("**الطبقة 0 — المدخل**: دفعة `x` بشكل `(B, 3)`: B ملاحظات، 3 خصائص. لا معلمات هنا."), action="Input", equation="x: (B, 3)"),
              Frame(_mlp_svg(1), caption("**الطبقة 1 — مخفية**: `z¹ = xW¹ + b¹` → `(B, 3)@(3, 5)+(5,) = (B, 5)` ثم `a¹ = ReLU(z¹)`. المعلمات: 3×5+5 = **20**."), action="Hidden", equation="a¹ = ReLU(xW¹ + b¹): (B, 5)", values=[("params", "", "20")]),
              Frame(_mlp_svg(2), caption("**الطبقة 2 — المخرج**: `z² = a¹W² + b²` → `(B, 5)@(5, 2)+(2,) = (B, 2)` ثم softmax → احتمالات لفئتين. المعلمات: 5×2+2 = **12**. المجموع 32."), action="Output", equation="p = softmax(a¹W² + b²): (B, 2)", values=[("params", "20", "32")])]
    animation_player("w04_flow", frames, title_ar="دفعة تعبر MLP: 3 → 5 → 2", stages=["Input", "Hidden", "Output"], interval_ms=1800)
    equation(r"a^{(\ell)} = f\big(a^{(\ell-1)} W^{(\ell)} + b^{(\ell)}\big), \qquad a^{(0)} = x, \qquad W^{(\ell)} \in \mathbb{R}^{n_{\ell-1}\times n_\ell}",
             [(r"a^{(\ell-1)}", "مخرج الطبقة السابقة بشكل (B, n_{ℓ−1})."), (r"W^{(\ell)}", "أوزان الطبقة: (وحدات سابقة × وحدات حالية)."), (r"b^{(\ell)}", "انحياز لكل وحدة، يُبث على الدفعة (يُضاف لكل صف)."), ("f", "تنشيط غير خطي (ReLU في المخفية؛ sigmoid/softmax/بلا في المخرج).")],
             meaning_ar="معادلة واحدة تتكرر لكل طبقة. بُعد الدفعة B لا يمسّه شيء؛ آخر بُعد يتغير إلى عدد وحدات الطبقة.",
             example_ar="(B,3) → (B,5) → (B,2): المعلمات 20 + 12 = 32.", dl_link_ar="`Dense(5, activation='relu')` هي هذه المعادلة بالضبط مع W بشكل (in, 5).", title_ar="معادلة الطبقة")
    compare_table(["المخرج", "التنشيط الأخير", "شكل المخرج", "الخسارة"],
                  [("انحدار (رقم واحد)", "بلا (linear)", "(B, 1)", "MSE"), ("تصنيف ثنائي", "sigmoid", "(B, 1)", "binary cross-entropy"),
                   ("تصنيف K فئات", "softmax", "(B, K)", "categorical / sparse categorical cross-entropy"), ("عدة أرقام معًا", "بلا", "(B, m)", "MSE")],
                  ["rtl", "ltr", "ltr", "ltr"])

    # ------------------------------------------------------------------ universal approximation
    h2("لماذا تكفي طبقة مخفية واحدة نظريًا؟ التقريب الشامل", "Why one hidden layer is enough in theory: universal approximation")
    st.markdown("منحنى غير خطي (طلب مقابل سعر — دالة تعليمية). كل وحدة ReLU تضيف **مفصلًا** `kink` واحدًا؛ المخرج يجمع الخطوط المكسورة. شاهد الخطأ ينهار مع إضافة الوحدات:")
    ad = relu_approx()
    acaps = []
    for it in ad["items"]:
        k = it["k"]
        if k == 1:
            acaps.append(f"**وحدة واحدة**: خط مستقيم تقريبًا — أفضل ما يستطيعه نموذج خطي. MSE = {it['mse']:.4f}.")
        elif k < 8:
            acaps.append(f"**{k} وحدات**: {k - 1} مفاصل تسمح للخط بتغيير ميله {k - 1} مرات. MSE = {it['mse']:.4f}. عدد المعلمات ≈ {it['n_params']}.")
        else:
            acaps.append(f"**{k} وحدة**: المنحنى شبه مطابق (MSE = {it['mse']:.4f}). بوحدات كافية يقترب الخطأ من الصفر لأي منحنى متصل على مجال محدود — هذه **مبرهنة التقريب الشامل**.")
    animation_player("w04_approx", [Frame(approx_svg(ad, i), caption(c), action=f"{ad['items'][i]['k']} units", values=[("MSE", "", f"{ad['items'][i]['mse']:.4f}")])
                                    for i, c in enumerate(acaps)], title_ar="منحنى من مفاصل ReLU", interval_ms=2000)
    equation(r"\hat y(x) = c + \sum_{j=1}^{k} v_j\,\operatorname{ReLU}(w_j x + b_j)",
             [(r"k", "عدد الوحدات المخفية (العرض)."), (r"\operatorname{ReLU}(w_j x + b_j)", "خط مكسور بمفصل عند x = −b_j/w_j."), (r"v_j", "وزن المخرج: كم يتغير الميل بعد المفصل.")],
             meaning_ar="شبكة بطبقة مخفية واحدة = دالة خطية قطعيًا. بمفاصل كافية تقرّب أي منحنى متصل.",
             example_ar=f"بـ 5 وحدات: MSE = {ad['items'][3]['mse']:.4f}؛ بـ 16 وحدة: {ad['items'][-1]['mse']:.5f}.",
             dl_link_ar="`Sequential([Dense(k, 'relu'), Dense(1)])` على مدخل واحد هي هذه المعادلة حرفيًا.", title_ar="الشبكة كخط مكسور")
    math_note("المبرهنة تقول إن التقريب **ممكن**، لا أن التدريب سيجده، ولا كم بيانات تحتاج. عمليًا: الشبكات الأعمق تحقق نفس الدقة بوحدات أقل بكثير لبعض الدوال (تركيب المفاصل يضاعفها)، لذلك نستعمل العمق — الدرس التالي يقيس ذلك تجريبيًا.")

    # ------------------------------------------------------------------ code
    h2("التحقق بالكود: NumPy = Keras", "Verify in code: NumPy = Keras")
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
        st.button("معمل العصبون", icon=":material/science:", on_click=go, args=("labs.neuron_lab",), key="w04_lab_neuron")
        st.button("المكافئ في PyTorch (nn.Module)", icon=":material/swap_horiz:", on_click=go, args=("foundations.frameworks.pytorch.nn_module",), key="w04_go_pt")
    intuition("فكّر في كل طبقة كـ«إعادة وصف» للدفعة بعدد جديد من الأعمدة: 3 خصائص خام → 5 خصائص متعلَّمة → 2 درجة قرار. الصفوف (الملاحظات) لا تختلط أبدًا في MLP.")
    common_mistake("كتابة W بشكل (out, in) في NumPy ثم `X @ W`: خطأ شكل. Keras (in, out) و`x @ W`؛ PyTorch (out, in) و`x @ W.T`. الأشكال أولًا دائمًا.")
    common_mistake("تمرير ملاحظة واحدة بشكل `(3,)` إلى النموذج: Dense تتوقع دفعة `(B, 3)`. استعمل `x[None, :]` أو `x.reshape(1, -1)`.")
    quiz("w04.flow", [
        Q("دفعة (64, 20) عبر Dense(50) ثم Dense(3): شكل المخرج", ["(64, 3)", "(3,)", "(20, 3)"], 0, ""),
        Q("معلمات الشبكة السابقة:", ["1050 + 153 = 1203", "1000 + 150", "70"], 0, "20×50+50 + 50×3+3."),
        Q("طبقتان خطيتان متتاليتان بلا تنشيط…", ["أقوى من واحدة", "تساويان طبقة خطية واحدة", "لا تُدرَّبان"], 1, ""),
        Q("بُعد الدفعة عبر الطبقات…", ["يتغير كل طبقة", "ثابت", "يختفي"], 1, ""),
        Q("الخلية Z[2, 4] في XW تساوي…", ["الصف 4 من X · العمود 2 من W", "الصف 2 من X · العمود 4 من W", "X[2,4]·W[2,4]"], 1, "ملاحظة 2، وحدة 4."),
        Q("وحدة ReLU مخرجها 0 لعميل معين تعني…", ["الوحدة معطلة للأبد", "z سالب لهذا العميل فلا تساهم في قراره", "خطأ في الأوزان"], 1, ""),
        Q("كل وحدة ReLU في طبقة مخفية واحدة على مدخل واحد تضيف…", ["دائرة", "مفصلًا (تغيير ميل) واحدًا", "طبقة"], 1, ""),
        Q("(32, 7) @ W يعطي (32, 12). شكل W:", ["(12, 7)", "(7, 12)", "(32, 12)"], 1, "البعدان الداخليان يتطابقان."),
    ])
    takeaway("MLP = تكرار a = f(aW + b). كل خلية في XW = ملاحظة · وحدة. الدفعة ثابتة الصفوف؛ الأعمدة = وحدات الطبقة. المعلمات Σ(in×out+out). ReLU تقسم الفضاء إلى مناطق، ومجموعها يقرّب أي منحنى. بلا تنشيط تُطوى الطبقات.")
    lesson_footer(LESSON, ["عميل واحد عصبونًا عصبونًا (تحريك).", "ضرب المصفوفات خلية خلية (تحريك).", "التدفق المرئي والمعادلة وجدول رؤوس المخرج.", "التقريب الشامل بمفاصل ReLU (تحريك).", "NumPy = Keras بنفس الأوزان."])
