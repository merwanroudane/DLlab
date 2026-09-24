import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, definition, interpretation_note, intuition, math_note, takeaway, warning_note, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from content.course.w06_activations._viz import dead_relu_run, dead_svg, layer_grad_norms, saturation_svg, softmax_svg, vanish_svg
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
    objectives_ar=[
        "رسم الدوال ومشتقاتها جنبًا إلى جنب، ومشاهدة z يتحرك عبر مناطق التشبع (تحريك).",
        "رؤية تلاشي التدرج طبقة طبقة في شبكة من 8 طبقات (انتشار عكسي حقيقي) لـ sigmoid وtanh وReLU، واشتقاقه كحاصل ضرب.",
        "مشاهدة وحدات ReLU تموت أثناء تدريب بمعدل تعلم كبير، وإثبات ذلك في Keras وعلاجه بـ Leaky ReLU.",
        "حساب softmax خطوة بخطوة وفهم درجة الحرارة.",
    ],
    terms=["derivative", "gradient", "softmax", "sigmoid", "relu", "dead_relu", "vanishing_gradient", "activation_function", "chain_rule", "logit"],
    labs=["labs.activation_lab"],
    difficulty="intermediate",
    summary_ar="المشتقة هي ما يمر من التدرج إلى الخلف. sigmoid (≤ 0.25) وtanh تتشبعان فيتلاشى التدرج أسيًا مع العمق؛ ReLU تمرر 1 أو 0 (قد تموت)؛ Leaky/ELU تصلح الصفر. Softmax تحوّل logits إلى احتمالات.",
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

    # ------------------------------------------------------------------ functions & derivatives
    h2("الدوال ومشتقاتها", "Functions and derivatives")
    sel = st.multiselect("اختر دوالًا للرسم", list(FUNCS), default=["sigmoid", "tanh", "relu", "elu"], key="w06_sel")
    z = np.linspace(-6, 6, 400)
    c1, c2 = st.columns(2)
    colors = ["#2563EB", "#7C3AED", "#059669", "#DB2777", "#D97706", "#0891B2", "#EA580C"]
    for col, idx, title in ((c1, 0, "f(z)"), (c2, 1, "f′(z)")):
        fig = go.Figure()
        for name, color in zip(sel, colors):
            fig.add_trace(go.Scatter(x=z, y=FUNCS[name][idx](z), name=name, line=dict(color=color, width=2.5)))
        fig.update_layout(height=300, title=title, margin=dict(l=10, r=10, t=35, b=10), legend=dict(orientation="h"))
        with col:
            st.plotly_chart(fig, width="stretch", key=f"w06_fig_{idx}")
    table(["الدالة", "المدى", "f′ القصوى", "التشبع", "الصفر الميت", "الاستخدام النموذجي"],
          [("sigmoid", "(0, 1)", "0.25", "الطرفان", "لا", "مخرج ثنائي فقط"), ("tanh", "(−1, 1)", "1", "الطرفان", "لا", "RNN/LSTM داخليًا؛ مخفي قديم"), ("ReLU", "[0, ∞)", "1", "لا (للموجب)", "**نعم** (z<0)", "الافتراضي للمخفي"),
           ("Leaky ReLU", "(−∞, ∞)", "1", "لا", "لا (ميل α)", "عند ظهور وحدات ميتة"), ("ELU", "(−α, ∞)", "1", "سالب ناعم", "لا", "متوسط ≈ 0؛ أبطأ قليلًا"), ("SELU", "مقيَّس", "1.05", "سالب ناعم", "لا", "شبكات كثيفة عميقة بتهيئة lecun_normal"),
           ("Swish", "(−0.28, ∞)", "≈1.1", "لا", "لا", "بديل ReLU في نماذج حديثة"), ("Softmax", "احتمالات تجمع لـ 1", "—", "—", "—", "مخرج متعدد الفئات فقط")],
          ["ltr", "code", "num", "rtl", "rtl", "rtl"])

    h3("z يعبر المنحنى: ماذا يمر إلى الخلف؟", "z sweeps across: what passes back?")
    zs = [-6.0, -4.0, -2.0, -0.5, 0.0, 0.5, 2.0, 4.0, 6.0]
    scaps = []
    for z0 in zs:
        sg = 1 / (1 + math.exp(-z0))
        d_s, d_t, d_r = sg * (1 - sg), 1 - math.tanh(z0) ** 2, float(z0 > 0)
        tail = ("sigmoid وtanh **متشبعتان**: تقريبًا لا شيء يمر. ReLU " + ("تمرر 1 كاملًا." if z0 > 0 else "تمرر **صفرًا**: الوحدة صامتة لهذا المدخل.")) if abs(z0) >= 4 else \
               ("قرب الصفر: sigmoid في أفضل حالاتها (0.25 فقط!)، وtanh تمرر حتى 1." if abs(z0) < 1 else "بداية التشبع.")
        scaps.append(f"**z = {z0:+.1f}**: σ′ = {d_s:.4f}، tanh′ = {d_t:.4f}، ReLU′ = {d_r:.0f}. {tail}")
    animation_player("w06_sat", [Frame(saturation_svg(z0), caption(c), action=f"z = {z0:+.1f}") for z0, c in zip(zs, scaps)],
                     title_ar="قيمة المشتقة = حجم ما يعود من التدرج", interval_ms=1500)
    intuition("اقرأ رسم المشتقة لا الدالة: عند |z| > 4 مشتقة sigmoid < 0.02؛ ثماني طبقات كهذه تضرب التدرج في 0.02⁸. ReLU مشتقتها 1 للموجب — لا تضعف الإشارة — لكنها 0 للسالب: إما تمر كاملة أو لا تمر.")

    # ------------------------------------------------------------------ vanishing gradients
    h2("تلاشي التدرج طبقة طبقة", "Vanishing gradients, layer by layer")
    st.markdown("شبكة من 8 طبقات مخفية (عرض 16) ومخرج، دفعة واحدة من 64 ملاحظة، **انتشار عكسي حقيقي** بمحرك NumPy. نقيس معيار تدرج أوزان كل طبقة، بدءًا من المخرج (يمين) نحو المدخل (يسار):")
    g = layer_grad_norms()
    L = len(g["sigmoid"])
    vcaps = []
    for upto in range(1, L + 1):
        layer = L - upto + 1
        vcaps.append(f"**الطبقة {layer}**: ‖∇W‖ — sigmoid {g['sigmoid'][layer - 1]:.1e}، tanh {g['tanh'][layer - 1]:.1e}، ReLU {g['relu'][layer - 1]:.1e}."
                     + (" كل خطوة إلى الخلف تضرب تدرج sigmoid في مشتقة ≤ 0.25 (ووزن) فيصغر هنا نحو 4–5 مرات في كل طبقة." if upto == 3 else "")
                     + (f" **الطبقة الأولى تتلقى تدرجًا أصغر بـ {g['sigmoid'][-1] / g['sigmoid'][0]:,.0f} مرة من المخرج مع sigmoid** — تكاد لا تتعلم. tanh وReLU (بتهيئة مناسبة) تحافظان على الحجم." if upto == L else ""))
    animation_player("w06_vanish", [Frame(vanish_svg(g, upto), caption(c), action=f"layer {L - upto + 1}") for upto, c in zip(range(1, L + 1), vcaps)],
                     title_ar="التدرج يعود من المخرج إلى المدخل (مقياس لوغاريتمي)", interval_ms=1300)
    equation(r"\frac{\partial L}{\partial W^{(1)}} \;\propto\; \prod_{\ell=2}^{L} W^{(\ell)}\, f'\!\big(z^{(\ell)}\big)",
             [(r"f'(z^{(\ell)})", "مشتقة التنشيط في الطبقة ℓ — ≤ 0.25 لـ sigmoid."), (r"W^{(\ell)}", "أوزان الطبقة؛ التهيئة تحدد حجمها."), (r"\prod", "حاصل ضرب عبر كل الطبقات فوق الطبقة الأولى.")],
             meaning_ar="التدرج الذي يصل إلى الطبقة الأولى حاصل ضرب L عاملًا. عوامل < 1 ⇒ تلاشٍ أسي؛ عوامل > 1 ⇒ انفجار أسي. ReLU مع تهيئة He تبقي العامل ≈ 1.",
             example_ar=f"sigmoid: 0.25⁸ ≈ {0.25 ** 8:.1e} (سقف نظري، قبل الأوزان). المقاس هنا: نسبة الأولى/الأخيرة = {g['sigmoid'][0] / g['sigmoid'][-1]:.1e}.",
             dl_link_ar="لهذا الشبكات العميقة تستعمل ReLU + تهيئة He (`kernel_initializer='he_normal'`)، وBatchNorm، واتصالات تخطٍّ في النماذج الحديثة.", title_ar="التدرج حاصل ضرب")
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

    # ------------------------------------------------------------------ dead ReLU in action
    h2("ReLU الميت أثناء التدريب", "Dead ReLU during training")
    st.markdown("شبكة 2 → 10 → 1 بـ ReLU على الهلالين، مرتين: η صغير وη كبير جدًا (SGD). كل خط = حد وحدة مخفية (حيث z = 0). **الخط الرمادي المتقطع = وحدة ميتة**: كل نقاط البيانات في جهتها السالبة.")
    dd = dead_relu_run()
    names = list(dd["runs"])
    steps_ = [s["step"] for s in dd["runs"][names[0]]]
    dcaps = []
    for i, st_ in enumerate(steps_):
        a, b = dd["runs"][names[0]][i], dd["runs"][names[1]][i]
        dcaps.append(f"**الخطوة {st_}**: η الصغير — {a['dead']} وحدات ميتة، دقة {a['acc']:.2f}. η الكبير — **{b['dead']} وحدات ميتة**، دقة {b['acc']:.2f}."
                     + (" خطوة تحديث ضخمة دفعت انحياز بعض الوحدات بعيدًا في السالب: خطوطها خرجت من منطقة البيانات، وتدرجها صفر، فلن تعود أبدًا." if b["dead"] and i >= 1 else ""))
    animation_player("w06_dead", [Frame(dead_svg(dd, i), caption(c), action=f"step {s}") for i, (s, c) in enumerate(zip(steps_, dcaps))],
                     title_ar="وحدات تموت بمعدل تعلم كبير (تدريب حقيقي)", interval_ms=1800)
    warning_note("الوحدة الميتة خسارة صامتة للسعة: الشبكة ذات 10 وحدات تعمل فعليًا بـ 6. لا رسالة خطأ؛ فقط أداء أسوأ وتذبذب. الوقاية: η معقول، تهيئة He، أو Leaky ReLU/ELU التي تُبقي ميلًا صغيرًا للسالب.")

    # ------------------------------------------------------------------ softmax
    h2("Softmax خطوة بخطوة", "Softmax step by step")
    why("Softmax ليست تنشيطًا «عنصريًا»: كل مخرج يعتمد على كل z في الطبقة (القسمة على المجموع). لذلك لا تُستخدم في الطبقات المخفية، ولا مع وحدة واحدة (تعطي 1 دائمًا).")
    zlog = [2.0, 1.0, 0.1]
    T = st.select_slider("درجة الحرارة T", options=[0.5, 1.0, 2.0, 5.0], value=1.0, key="w06_T")
    zt = [v / T for v in zlog]
    e = [math.exp(v - max(zt)) for v in zt]
    p = [v / sum(e) for v in e]
    fcaps = [f"**logits**: ثلاث درجات خام لعميل (سداد، تأخر، تعثر): {zlog}" + (f"، مقسومة على T = {T:g}." if T != 1 else ". يمكن أن تكون سالبة وأي حجم."),
             "**الأُسّ**: `e^(z − max z)` يجعل كل شيء موجبًا. طرح الأكبر لا يغيّر النتيجة (يُختصر في القسمة) لكنه يمنع فيضان الأعداد — هذا ما تفعله المكتبات.",
             f"**المجموع**: {sum(e):.3f} — المقام المشترك.",
             f"**القسمة**: كل أس ÷ المجموع ⇒ احتمالات: {', '.join(f'{v:.3f}' for v in p)}. المجموع 1 بالضبط؛ الترتيب نفسه ترتيب logits."]
    animation_player(f"w06_softmax_{T}", [Frame(softmax_svg(zlog, i, T), caption(c), action=["logits", "exp", "sum", "normalize"][i]) for i, c in enumerate(fcaps)],
                     title_ar="من logits إلى احتمالات", interval_ms=2000)
    equation(r"p_k = \frac{e^{z_k / T}}{\sum_{j=1}^{K} e^{z_j / T}}",
             [(r"z_k", "logit الفئة k (مخرج Dense بلا تنشيط)."), ("T", "درجة الحرارة: T > 1 تسطّح التوزيع، T < 1 تحدّه."), ("K", "عدد الفئات.")],
             meaning_ar="تحويل درجات بلا حدود إلى احتمالات موجبة مجموعها 1، مع الحفاظ على الترتيب. الفروق بين logits هي المهمة لا قيمها.",
             example_ar=f"T = {T:g}: p = [{', '.join(f'{v:.3f}' for v in p)}]. جرّب T = 5 (شبه منتظم) وT = 0.5 (شبه قرار حاد).",
             dl_link_ar="`Dense(K, activation='softmax')`؛ أو `Dense(K)` + `from_logits=True` في الخسارة (أكثر استقرارًا عدديًا).", title_ar="Softmax")
    math_note("مع فئتين، softmax على logits (z₁، z₂) = sigmoid(z₁ − z₂). لذلك التصنيف الثنائي بوحدة sigmoid واحدة يكافئ softmax بوحدتين — بمعلمات أقل.")
    common_mistake("`Dense(1, activation='softmax')`: مخرج ثابت = 1.0 لكل ملاحظة، والخسارة لا تنخفض. الثنائي = sigmoid بوحدة واحدة أو softmax بوحدتين.")
    quiz("w06.fn", [
        Q("مشتقة sigmoid عند z = 5 ≈", ["0.25", "0.0066", "1"], 1, "تشبع: σ(5)(1 − σ(5))."),
        Q("ReLU الميت:", ["z > 0 دائمًا", "z < 0 لكل المدخلات → تدرج 0", "مشتقة كبيرة"], 1, ""),
        Q("أي دالة تحافظ على معيار التدرج عبر 8 طبقات؟", ["sigmoid", "ReLU / ELU (مع تهيئة مناسبة)", "softmax"], 1, ""),
        Q("Softmax في طبقة مخفية…", ["ممتازة", "غير مناسبة: ليست عنصرية وتُطبّع الطبقة", "تسرّع"], 1, ""),
        Q("التدرج الذي يصل إلى الطبقة الأولى هو…", ["مجموع", "حاصل ضرب عوامل عبر الطبقات", "متوسط"], 1, ""),
        Q("سبب شائع لموت وحدات ReLU أثناء التدريب:", ["η صغير", "η كبير جدًا (قفزة تدفع الانحياز للسالب)", "بيانات كثيرة"], 1, "رأيته في التحريك."),
        Q("رفع درجة الحرارة T في softmax…", ["يحدّ التوزيع", "يسطّح التوزيع", "لا يغيّر شيئًا"], 1, ""),
        Q("softmax([3, 3, 3]) =", ["[1, 1, 1]", "[1/3, 1/3, 1/3]", "[0, 0, 1]"], 1, ""),
    ])
    takeaway("المشتقة هي ما يمر من التدرج؛ التدرج في العمق حاصل ضرب. sigmoid/tanh تتشبعان (sigmoid تتلاشى أسيًا مع العمق)؛ ReLU تمرر 1 أو 0 وقد تموت بـ η كبير؛ Leaky/ELU تصلح الصفر. Softmax = أس ÷ مجموع للمخرج المتعدد فقط.")
    lesson_footer(LESSON, ["الدوال والمشتقات بالرسم والجدول.", "z يعبر مناطق التشبع (تحريك).", "تلاشي التدرج طبقة طبقة (تحريك) وحاصل الضرب.", "التلاشي وReLU الميت بالأرقام في Keras.",
                           "وحدات تموت أثناء التدريب (تحريك).", "Softmax خطوة بخطوة ودرجة الحرارة."])
