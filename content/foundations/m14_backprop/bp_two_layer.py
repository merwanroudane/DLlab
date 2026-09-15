import numpy as np
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, intuition, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import worked_steps
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.backprop.backpropagation.two_layer",
    title_ar="مثال بطبقتين: مصفوفات وأرقام وتحريك أمامي/خلفي",
    title_en="Two-Layer Example: Matrices, Numbers & a Forward/Backward Animation",
    module="foundations.backprop",
    parent="foundations.backprop.backpropagation",
    order=4,
    prerequisites=["foundations.backprop.backpropagation.one_neuron"],
    objectives_ar=["تطبيق قواعد δ على شبكة 2 → 3 → 1 بدفعة من ملاحظتين، بالأرقام.", "مشاهدة كل خطوة خلفية في تحريك مع الأشكال والقيم."],
    terms=["backpropagation", "hadamard_product"],
    difficulty="intermediate",
    summary_ar="شبكة 2 → 3 → 1: أمامي على دفعة (2, 2)، ثم δ₂ = (p − y)/n، gW₂ = a₁ᵀδ₂، δ₁ = (δ₂W₂ᵀ) ⊙ ReLU′(z₁)، gW₁ = Xᵀδ₁.",
)

X = np.array([[1.0, 2.0], [-1.0, 0.5]]); Y = np.array([1.0, 0.0])
W1 = np.array([[0.5, -1.0, 0.2], [0.3, 0.8, -0.5]]); B1 = np.array([0.1, 0.0, -0.2])
W2 = np.array([[1.0], [-1.5], [0.7]]); B2 = np.array([0.3])


def _fmt(a):
    return np.array2string(np.asarray(a), precision=3, suppress_small=True)


def render() -> None:
    lesson_header(LESSON)
    h2("الحساب الكامل", "The full computation")
    z1 = X @ W1 + B1; a1 = np.maximum(0, z1); z2 = a1 @ W2 + B2; p = 1 / (1 + np.exp(-z2)); n = 2
    L = -np.mean(Y * np.log(p[:, 0]) + (1 - Y) * np.log(1 - p[:, 0]))
    d2 = (p[:, 0] - Y)[:, None] / n; gW2 = a1.T @ d2; gb2 = d2.sum(0)
    d1 = (d2 @ W2.T) * (z1 > 0); gW1 = X.T @ d1; gb1 = d1.sum(0)
    st.code(f"X (2,2) =\n{_fmt(X)}\ny = {Y}\nW1 (2,3) =\n{_fmt(W1)}\nb1 = {_fmt(B1)}\nW2 (3,1) =\n{_fmt(W2)}\nb2 = {_fmt(B2)}", language="text")
    intuition("دفعة من ملاحظتين: كل مصفوفة وسيطة لها صفان. الملاحظة الثانية لها هدف 0 والأولى 1، فستحصلان على δ بإشارتين مختلفتين.")
    frames = [
        Frame(f"<pre style='text-align:left'>z1 = X@W1 + b1 =\n{_fmt(z1)}\na1 = ReLU(z1) =\n{_fmt(a1)}</pre>", caption("**أمامي 1/2** — الطبقة المخفية: `z1 (2,3)` ثم ReLU. لاحظ الصفر في الصف الأول العمود الثاني (z سالب)."), action="Forward", highlight=0),
        Frame(f"<pre style='text-align:left'>z2 = a1@W2 + b2 =\n{_fmt(z2)}\np = sigmoid(z2) =\n{_fmt(p)}\nL = {L:.4f}</pre>", caption(f"**أمامي 2/2** — الإخراج واحتمالان، ثم BCE على الدفعة: `L = {L:.4f}`."), action="Forward", highlight=0),
        Frame(f"<pre style='text-align:left'>delta2 = (p - y)/n =\n{_fmt(d2)}</pre>", caption("**خلفي 1/4** — δ₂ = (p − y)/n بالشكل `(2,1)`: الملاحظة الأولى (y=1) سالبة، الثانية (y=0) موجبة."), action="Backward", equation="delta2 = (p - y) / n", highlight=1),
        Frame(f"<pre style='text-align:left'>gW2 = a1.T @ delta2 =\n{_fmt(gW2)}   shape {gW2.shape}\ngb2 = sum(delta2) = {_fmt(gb2)}</pre>", caption("**خلفي 2/4** — تدرج الإخراج: `a1.T (3,2) @ delta2 (2,1) = (3,1)` يطابق شكل W2."), action="Backward", equation="gW2 = a1ᵀ δ2 ;  gb2 = Σ δ2", highlight=1),
        Frame(f"<pre style='text-align:left'>delta2 @ W2.T =\n{_fmt(d2 @ W2.T)}\nReLU'(z1) =\n{_fmt((z1 > 0).astype(int))}\ndelta1 = product =\n{_fmt(d1)}</pre>", caption("**خلفي 3/4** — العودة إلى المخفية: `delta2 @ W2.T (2,3)` ثم هادامارد مع مشتقة ReLU. **الخلية الميتة (0) تقطع التدرج**."), action="Backward", equation="delta1 = (δ2 W2ᵀ) ⊙ ReLU′(z1)", highlight=1),
        Frame(f"<pre style='text-align:left'>gW1 = X.T @ delta1 =\n{_fmt(gW1)}   shape {gW1.shape}\ngb1 = sum(delta1) = {_fmt(gb1)}</pre>", caption("**خلفي 4/4** — تدرج الطبقة الأولى: `X.T (2,2) @ delta1 (2,3) = (2,3)` يطابق W1. انتهى الانتشار الخلفي؛ المحسّن يتولى التحديث."), action="Backward", equation="gW1 = Xᵀ δ1 ;  gb1 = Σ δ1", values=[("|gW1|", "", f"{np.linalg.norm(gW1):.4f}"), ("|gW2|", "", f"{np.linalg.norm(gW2):.4f}")], highlight=1),
    ]
    animation_player("bp2_anim", frames, title_ar="أمامي ثم خلفي على شبكة 2 → 3 → 1", stages=["Forward", "Backward"], interval_ms=2500)
    table(["الكمية", "الشكل", "القاعدة"],
          [("δ₂", "(2, 1)", "(p − y)/n"), ("gW₂", "(3, 1)", "a₁ᵀ δ₂"), ("gb₂", "(1,)", "Σ δ₂"), ("δ₁", "(2, 3)", "(δ₂ W₂ᵀ) ⊙ ReLU′(z₁)"), ("gW₁", "(2, 3)", "Xᵀ δ₁"), ("gb₁", "(3,)", "Σ δ₁")],
          ["code", "code", "code"])
    common_mistake("ملاحظة الخلية الميتة: عمود δ₁ الثاني للملاحظة الأولى = 0 لأن ReLU′ = 0 هناك. أوزان تلك الخلية لا تتعلم من هذه الملاحظة — وهذا طبيعي لملاحظة واحدة، وكارثي إن حدث لكل الملاحظات (Dead ReLU).")
    quiz("bp.two", [
        Q("شكل δ₁ لدفعة من 2 وطبقة مخفية من 3…", ["(3, 1)", "(2, 3)", "(3, 2)"], 1, "(n, units).", kind="shape"),
        Q("لماذا يظهر W₂ᵀ في حساب δ₁؟", ["للسرعة", "لتوزيع خطأ الإخراج على مدخلاته عبر نفس الأوزان", "لتغيير الشكل فقط"], 1, "الطريق نفسه معكوسًا."),
        Q("ReLU′ = 0 لخلية في ملاحظة ما يعني…", ["تدرج أوزان تلك الخلية من تلك الملاحظة = 0", "خطأ في الكود", "الخسارة صفر"], 0, "البوابة مغلقة."),
    ])
    takeaway("طبقتان = تطبيق القواعد مرتين. الأشكال تتطابق مع المعلمات. مشتقة ReLU تعمل كبوابة.")
    lesson_footer(LESSON, ["أرقام حقيقية لكل δ وتدرج.", "التحريك يربط كل خطوة بمعادلتها.", "الخلية الميتة تقطع التدرج."])
