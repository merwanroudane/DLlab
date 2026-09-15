import numpy as np
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway
from components.comparison import compare_table
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table
from labs.fw import keras_summary_info

LESSON = Lesson(
    id="foundations.frameworks.keras.summary_deconstruction",
    title_ar="تفكيك model.summary() بصريًا وعدّ المعلمات",
    title_en="model.summary() Visual Deconstruction & Parameter Count",
    module="foundations.frameworks",
    parent="foundations.frameworks.keras",
    order=9,
    prerequisites=["foundations.frameworks.keras.layers_models", "foundations.architecture.parameter_count"],
    objectives_ar=["قراءة كل عمود في summary: اسم الطبقة ونوعها، شكل المخرج، عدد المعلمات.", "لكل طبقة: شكل المدخل، شكل المخرج، كيف حُسبت المعلمات، والمعادلة.", "الفرق بين المعلمات القابلة للتدريب وغير القابلة (BN) ومعلمات المحسّن."],
    terms=["parameter", "shape", "batch_dimension"],
    labs=["labs.parameter_counter"],
    difficulty="beginner",
    summary_ar="summary = جدول (طبقة، شكل المخرج، معلمات). None = بُعد الدفعة. Dense: in×units + units. BN: 4 لكل خاصية (2 متعلَّمة + 2 متحركة غير قابلة للتدريب). Dropout/Flatten: 0.",
)


def _spec_from_ui():
    st.markdown("**ابنِ النموذج** (كل صف طبقة):")
    c0, c1, c2, c3 = st.columns(4)
    with c0:
        n_in = st.select_slider("خصائص المدخل", options=[2, 4, 8, 20, 784], value=4, key="sd_in")
    with c1:
        h1 = st.select_slider("Dense 1 units", options=[4, 8, 16, 32, 64, 128], value=16, key="sd_h1")
    with c2:
        use_bn = st.toggle("BatchNormalization بعدها", value=True, key="sd_bn")
    with c3:
        drop = st.select_slider("Dropout", options=[0.0, 0.2, 0.5], value=0.2, key="sd_do")
    c4, c5 = st.columns(2)
    with c4:
        h2_ = st.select_slider("Dense 2 units (0 = بدون)", options=[0, 8, 16, 32], value=8, key="sd_h2")
    with c5:
        out = st.select_slider("مخرجات", options=[1, 3, 10], value=3, key="sd_out")
    spec = [("dense", h1, "relu")]
    if use_bn:
        spec.append(("bn",))
    if drop > 0:
        spec.append(("dropout", drop))
    if h2_ > 0:
        spec.append(("dense", h2_, "relu"))
    spec.append(("dense", out, "softmax" if out > 1 else "sigmoid"))
    return tuple(spec), (int(n_in),)


def _explain_layer(r: dict) -> None:
    fin = r["input"][-1]; fout = r["output"][-1]
    st.markdown(f"**شكل المدخل** `{r['input']}` → **شكل المخرج** `{r['output']}` — `None` هو بُعد الدفعة: الطبقة تقبل أي عدد من الملاحظات.")
    if r["type"] == "Dense":
        act = r["config"].get("activation") or "none"
        equation(rf"\mathbf{{a}} = f(\mathbf{{x}}\,W + \mathbf{{b}}), \qquad W \in \mathbb{{R}}^{{{fin}\times{fout}}},\; \mathbf{{b}} \in \mathbb{{R}}^{{{fout}}}",
                 [(rf"W", f"kernel بشكل ({fin}, {fout}): {fin}×{fout} = {fin * fout} وزنًا."), (r"\mathbf{b}", f"bias بشكل ({fout},): {fout} انحيازًا."), ("f", f"التنشيط: {act}.")],
                 meaning_ar=f"عدد المعلمات = {fin}×{fout} + {fout} = **{r['params']}**. كلها قابلة للتدريب.",
                 example_ar=f"دفعة (32, {fin}) × W ({fin}, {fout}) → (32, {fout}) ثم + b بالبث.", dl_link_ar="هذا السطر في summary هو الوحدة 10 بالضبط.", title_ar=f"{r['name']} — Dense({fout})")
    elif r["type"] == "BatchNormalization":
        equation(rf"\hat z_j = \frac{{z_j - \mu_j}}{{\sqrt{{\sigma^2_j + \epsilon}}}}, \qquad y_j = \gamma_j \hat z_j + \beta_j, \qquad j = 1..{fout}",
                 [(r"\gamma, \beta", f"متعلَّمتان: 2×{fout} = {r['trainable']} معلمة قابلة للتدريب."), (r"\mu, \sigma^2", f"متوسطات متحركة: 2×{fout} = {r['non_trainable']} معلمة **غير قابلة للتدريب** (تُحدَّث بالإحصاء لا بالتدرج).")],
                 meaning_ar=f"المجموع {r['params']} = 4×{fout}. summary يعدّها كلها في Param # ويفصلها في السطر الأخير Trainable/Non-trainable.",
                 example_ar="Non-trainable params ≠ 0 في نموذج بلا طبقات مجمّدة = يوجد BN.", dl_link_ar="وحدة 20: BN.", title_ar=f"{r['name']} — BatchNormalization")
    elif r["type"] == "Dropout":
        st.markdown(f"**Dropout(rate={r['config'].get('rate')})**: لا معلمات (0). في التدريب يصفّر كل وحدة باحتمال rate ويقسّم الباقي على (1−rate)؛ في الاستدلال هوية. الشكل لا يتغير: `{r['input']}` → `{r['output']}`.")
    elif r["type"] == "Flatten":
        st.markdown(f"**Flatten**: لا معلمات. يعيد ترتيب `{r['input']}` إلى `{r['output']}` (ضرب الأبعاد عدا الدفعة).")
    if r["weights"]:
        table(["الوزن", "الشكل", "العناصر"], [(n, str(s), str(int(np.prod(s)))) for n, s in r["weights"]], ["code", "code", "num"])


def render() -> None:
    lesson_header(LESSON)
    definition("**`model.summary()`**: جدول بثلاثة أعمدة — **Layer (type)**: اسم الطبقة (تلقائي مثل `dense_1` أو ما سمّيته) ونوعها؛ **Output Shape**: شكل المخرج مع `None` لبُعد الدفعة؛ **Param #**: عدد المعلمات في الطبقة. ثم ثلاثة أسطر: Total، Trainable، Non-trainable (وربما Optimizer params بعد التدريب).")
    spec, in_shape = _spec_from_ui()
    info = keras_summary_info(spec, in_shape)
    h2("النص كما يطبعه Keras", "The raw text")
    st.code(info["summary"].strip(), language="text")
    h2("نفس الجدول مفكّكًا", "The same table, deconstructed")
    rows = [(r["name"], r["type"], str(r["input"]), str(r["output"]), str(r["params"]), str(r["trainable"]), str(r["non_trainable"])) for r in info["rows"]]
    table(["Layer name", "Layer type", "Input shape", "Output shape", "Param #", "Trainable", "Non-trainable"], rows, ["code", "code", "code", "code", "num", "num", "num"])
    st.markdown(f"**Total params** = مجموع العمود = **{info['total']}** · Trainable = **{info['trainable']}** · Non-trainable = **{info['non_trainable']}**")
    h3("اضغط على طبقة", "Click a layer")
    names = [r["name"] for r in info["rows"]]
    sel = st.segmented_control("الطبقة", options=names, default=names[0], key="sd_layer", label_visibility="collapsed")
    r = next((x for x in info["rows"] if x["name"] == sel), info["rows"][0])
    with st.container(border=True):
        _explain_layer(r)
    intuition("summary لا يعرف **شكل المدخل** لكل طبقة بشكل مباشر — هو شكل مخرج الطبقة السابقة. لذلك لقراءة Param # لطبقة Dense تحتاج السطر الذي فوقها: `in` من فوق، `units` من السطر نفسه.")
    h2("قواعد العدّ السريعة", "Quick counting rules")
    compare_table(["الطبقة", "المعلمات", "قابلة للتدريب؟"],
                  [("Dense(units) بمدخل in", "in × units + units", "نعم"), ("BatchNormalization على f خاصية", "4f (γ, β + μ, σ² متحركتان)", "2f نعم، 2f لا"), ("Dropout / Flatten / Activation / Input", "0", "—"),
                   ("Conv2D(k, (h, w)) بقنوات c_in", "h·w·c_in·k + k", "نعم (الأسبوع 8)"), ("Embedding(vocab, d)", "vocab × d", "نعم (الأسبوع 10)"), ("طبقة مجمّدة (`trainable=False`)", "كما هي، لكن تُعدّ Non-trainable", "لا")],
                  ["rtl", "code", "rtl"])
    common_mistake("«Non-trainable params ≠ 0 يعني خطأ». لا: BN تضيف 2 لكل خاصية غير قابلة للتدريب بالتصميم، والطبقات المجمّدة في النقل التعلّمي كذلك. الخطأ هو أن تكون **كل** المعلمات غير قابلة للتدريب دون قصد (`model.trainable = False` منسي).")
    quiz("keras.summary", [
        Q("`(None, 16)` في Output Shape…", ["16 ملاحظة", "أي عدد من الملاحظات × 16 خاصية", "16 طبقة"], 1, "None = الدفعة."),
        Q("Dense(32) بعد طبقة مخرجها (None, 20): Param #", ["640", "660", "52"], 1, "20×32 + 32."),
        Q("BatchNormalization على 64 خاصية: Non-trainable", ["0", "64", "128"], 2, "μ وσ² لكل خاصية."),
        Q("لحساب معلمات Dense من summary تحتاج…", ["سطرها فقط", "سطرها وشكل مخرج السطر السابق", "Total params"], 1, "in من الأعلى."),
    ])
    takeaway("summary = (اسم/نوع، شكل المخرج، معلمات). اقرأ كل سطر مع السطر السابق. Dense: in×units+units. BN: 4f نصفها غير قابل للتدريب. Dropout 0. تحقق من Total يدويًا مرة واحدة على الأقل لكل نموذج.")
    lesson_footer(LESSON, ["الأعمدة الثلاثة.", "تفكيك تفاعلي لكل طبقة.", "قواعد العدّ."])
