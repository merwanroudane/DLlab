import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import intuition, takeaway
from components.diagnostics import Problem, problem_card
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.backprop.backpropagation.vanishing_exploding",
    title_ar="تلاشي التدرج وانفجاره",
    title_en="Vanishing & Exploding Gradients",
    module="foundations.backprop",
    parent="foundations.backprop.backpropagation",
    order=6,
    prerequisites=["foundations.backprop.backpropagation.implementation", "foundations.activations.derivatives_saturation_dead"],
    objectives_ar=["رؤية بالمحاكاة كيف يتناقص أو ينفجر معيار التدرج مع العمق حسب التنشيط ومقياس الأوزان.", "الإطار الكامل للمشكلتين: أعراض، أسباب، تشخيص، إصلاح."],
    terms=["gradient", "norm"],
    labs=["labs.activation_lab", "labs.chain_rule_lab"],
    difficulty="intermediate",
    summary_ar="حاصل ضرب (Wᵀ ⊙ f′) عبر الطبقات إما يتقلص (تلاشٍ) أو يتضخم (انفجار). التنشيط والتهيئة والتطبيع والقصّ هي الأدوات.",
)


def _simulate(depth: int, width: int, act: str, scale: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    f = {"sigmoid": lambda z: 1 / (1 + np.exp(-z)), "tanh": np.tanh, "relu": lambda z: np.maximum(0, z)}[act]
    df = {"sigmoid": lambda z: (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z))), "tanh": lambda z: 1 - np.tanh(z) ** 2, "relu": lambda z: (z > 0).astype(float)}[act]
    a = rng.normal(size=(128, width)); zs, Ws = [], []
    for _ in range(depth):
        W = rng.normal(0, scale / np.sqrt(width), (width, width)); z = a @ W; zs.append(z); Ws.append(W); a = f(z)
    delta = rng.normal(size=(128, width)) / 128
    norms = []
    for l in reversed(range(depth)):
        delta = (delta @ Ws[l].T) * df(zs[l]); norms.append(float(np.linalg.norm(delta)))
    return norms[::-1]


def render() -> None:
    lesson_header(LESSON)
    h2("المحاكاة", "Simulation")
    c1, c2, c3 = st.columns(3)
    with c1:
        act = st.selectbox("التنشيط", ["sigmoid", "tanh", "relu"], key="ve_act")
    with c2:
        scale = st.select_slider("مقياس الأوزان (× 1/√width)", options=[0.5, 1.0, 1.41, 2.0, 3.0], value=1.41, key="ve_scale")
    with c3:
        depth = st.slider("العمق", 2, 30, 12, key="ve_depth")
    norms = _simulate(depth, 32, act, scale)
    fig = go.Figure(go.Scatter(x=list(range(1, depth + 1)), y=norms, mode="lines+markers", line=dict(color="#C8473A", width=3)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="layer (1 = first)", yaxis=dict(title="‖δ‖ (log)", type="log"), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9")
    st.plotly_chart(fig, width="stretch", key="ve_fig")
    ratio = norms[0] / max(norms[-1], 1e-300)
    st.code(f"‖δ‖ at last layer = {norms[-1]:.3e}   at first layer = {norms[0]:.3e}   ratio first/last = {ratio:.3e}", language="text")
    intuition("Sigmoid بأي مقياس: المنحنى يهبط (تلاشٍ). ReLU بمقياس 1.41 (≈ He): مستوٍ تقريبًا. ReLU بمقياس 3: يصعد (انفجار). العمق يضاعف الأثر أسّيًا.")
    problem_card(Problem(
        key="vanishing_grad", name_ar="تلاشي التدرج", name_en="Vanishing gradients",
        description_ar="معيار التدرج يتناقص أسّيًا مع العودة إلى الطبقات الأولى فتتوقف عن التعلم.",
        symptoms_ar=["الطبقات الأولى لا تتغير؛ الأخيرة تتعلم قليلًا.", "خسارة تهبط ببطء شديد ثم تستقر مرتفعة.", "الشبكة العميقة أسوأ من الضحلة على نفس البيانات."],
        sees_ar=["معايير التدرج لكل طبقة: 1e-1 (أخيرة) → 1e-8 (أولى).", "هيستوغرام تنشيطات مشبعة.", "تدريب عميق يبدو «كأنه شبكة من طبقتين»."],
        possible_causes_ar=["Sigmoid/tanh في طبقات كثيرة.", "تهيئة أوزان صغيرة جدًا.", "طول تسلسل كبير في RNN (نفس الآلية عبر الزمن)."],
        root_causes_ar=["حاصل ضرب عوامل أصغر من 1 (f′ ≤ 0.25، أو ‖W‖ صغير) عبر طبقات كثيرة."],
        diagnosis_ar=["سجّل معيار التدرج لكل طبقة كل حقبة.", "قارن نسبة الأولى/الأخيرة: < 1e-3 تلاشٍ.", "جرّب ReLU + He بنفس البنية."],
        evidence_ar=["النسبة تعود إلى ~1 بعد الإصلاح؛ الطبقات الأولى تبدأ التغير."],
        fixes_ar=["ReLU وعائلته.", "تهيئة He/Glorot.", "Batch Normalization.", "وصلات متبقية (ResNet).", "LSTM/GRU بدل RNN بسيطة للتسلسلات."],
        misdiagnosis_ar=["رفع معدل التعلم (ينفجر آخر الطبقات)."],
        related_ar=["الإشباع", "الانفجار", "Dead ReLU"],
        checklist_ar=["التنشيط؟", "التهيئة؟", "معايير التدرج مسجلة؟"],
    ))
    problem_card(Problem(
        key="exploding_grad", name_ar="انفجار التدرج", name_en="Exploding gradients",
        description_ar="معيار التدرج يتضخم أسّيًا فتقفز المعلمات وتصبح الخسارة ضخمة أو NaN.",
        symptoms_ar=["الخسارة تقفز فجأة أو تصبح `inf`/`nan`.", "أوزان ضخمة بعد خطوات قليلة.", "تذبذب عنيف."],
        sees_ar=["`loss: nan` بعد حقبة أو خطوة.", "معيار التدرج 1e3+.", "تحذير overflow."],
        possible_causes_ar=["تهيئة كبيرة.", "معدل تعلم كبير (يفاقم).", "تسلسلات طويلة في RNN.", "مدخلات غير محجّمة."],
        root_causes_ar=["حاصل ضرب عوامل أكبر من 1 (‖W‖ كبير) عبر الطبقات/الزمن."],
        diagnosis_ar=["راقب معيار التدرج قبل كل تحديث.", "افحص التهيئة والتحجيم.", "خفّض معدل التعلم وأعد."],
        evidence_ar=["القصّ يوقف NaN فورًا؛ التهيئة الصحيحة تعيد المعيار إلى ~1."],
        fixes_ar=["قصّ التدرج `clipnorm`/`clip_grad_norm_`.", "تهيئة He/Glorot.", "معدل تعلم أصغر / warm-up.", "تحجيم المدخلات.", "Batch Normalization."],
        tradeoffs_ar=["القصّ يخفي المشكلة الجذرية أحيانًا؛ استخدمه مع الإصلاح لا بدله."],
        misdiagnosis_ar=["«البيانات فيها خطأ» بينما السبب التهيئة."],
        related_ar=["التلاشي", "معدل التعلم", "NaN"],
        checklist_ar=["معيار التدرج قبل التحديث؟", "clipnorm مفعّل؟", "المدخلات محجّمة؟"],
        challenge=[Q("خسارة 0.69 → 0.62 → 4.3 → nan. التشخيص؟", ["تلاشٍ", "انفجار (غالبًا معدل تعلم/تهيئة)", "تسريب"], 1, "قفزة ثم NaN.", kind="curve")],
    ))
    quiz("bp.ve", [
        Q("عامل < 1 مضروب 20 مرة…", ["ينفجر", "يتلاشى", "يبقى"], 1, "أسّي."),
        Q("أول إجراء عند `loss: nan`…", ["زيادة الحقب", "قصّ التدرج + فحص التهيئة ومعدل التعلم", "تغيير الخسارة"], 1, "الانفجار."),
    ])
    takeaway("العمق يضاعف أثر كل عامل أسّيًا: < 1 تلاشٍ، > 1 انفجار. ReLU + He + BN + قصّ هي الأدوات؛ معايير التدرج لكل طبقة هي المقياس.")
    lesson_footer(LESSON, ["المحاكاة تُظهر الاتجاه فورًا.", "مشكلتان بإطار كامل.", "سجّل معيار التدرج دائمًا."])
