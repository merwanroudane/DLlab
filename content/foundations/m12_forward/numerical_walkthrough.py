import numpy as np
import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, takeaway
from components.diagram import svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import worked_steps
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.forward.numerical_walkthrough",
    title_ar="مثال رقمي كامل مع تحريك: 2 → 3 → 1",
    title_en="Full Numerical Walkthrough with Animation: 2 → 3 → 1",
    module="foundations.forward",
    order=3,
    prerequisites=["foundations.forward.logits_probability_threshold"],
    objectives_ar=["تنفيذ تمرير أمامي كامل بالأرقام لشبكة صغيرة.", "مشاهدة كل قيمة تتحرك عبر الشبكة خطوة بخطوة."],
    terms=["weight", "bias", "softmax"],
    difficulty="beginner",
    summary_ar="شبكة 2 → 3 → 1 بأرقام فعلية: كل z وa يُحسب ويُعرض في تحريك.",
)

X = np.array([1.0, 2.0])
W1 = np.array([[0.5, -1.0, 0.2], [0.3, 0.8, -0.5]]); B1 = np.array([0.1, 0.0, -0.2])
W2 = np.array([[1.0], [-1.5], [0.7]]); B2 = np.array([0.3])


def _svg(hl: dict) -> str:
    s = '<svg viewBox="0 0 560 240" width="100%" style="max-width:560px">' + svg_defs()
    xs = [60, 280, 500]; ys = [[80, 160], [50, 120, 190], [120]]
    for li in range(2):
        for i, y1 in enumerate(ys[li]):
            for j, y2 in enumerate(ys[li + 1]):
                s += f'<line x1="{xs[li] + 22}" y1="{y1}" x2="{xs[li + 1] - 22}" y2="{y2}" stroke="#C9C1B3"/>'
    for li, layer in enumerate(ys):
        for i, y in enumerate(layer):
            key = f"{li}_{i}"; fill = "#1F7A78" if key in hl.get("active", []) else ["#E6F1FB", "#EFE9FA", "#DDF5EA"][li]
            s += f'<circle cx="{xs[li]}" cy="{y}" r="22" fill="{fill}" stroke="#6B675F"/>'
            val = hl.get("values", {}).get(key)
            s += svg_text(xs[li], y + 5, val if val is not None else ["x", "a", "ŷ"][li] + str(i + 1), size=12, bold=True, color="#fff" if key in hl.get("active", []) else "#2B2A28", mono=val is not None)
    s += svg_text(60, 225, "input", size=12) + svg_text(280, 225, "hidden (ReLU)", size=12) + svg_text(500, 225, "output (sigmoid)", size=12)
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("الشبكة والأرقام", "Network & numbers")
    st.code(f"x  = {X}\nW1 = {W1.tolist()}   b1 = {B1.tolist()}\nW2 = {W2[:, 0].tolist()}   b2 = {B2.tolist()}", language="text")
    z1 = X @ W1 + B1; a1 = np.maximum(0, z1); z2 = a1 @ W2 + B2; p = 1 / (1 + np.exp(-z2))
    intuition("ثلاث خلايا مخفية وخلية إخراج. سنحسب كل خلية على حدة؛ لاحظ أن الخلية المخفية الثانية تعطي z سالبًا فتصفّرها ReLU — وهذا يؤثر في كل ما بعدها.")
    frames = [Frame(_svg({"values": {"0_0": "1.0", "0_1": "2.0"}, "active": ["0_0", "0_1"]}), caption("**المدخل** `x = (1.0, 2.0)`: ملاحظة واحدة بخاصيتين."), action="Input")]
    for j in range(3):
        terms = " + ".join(f"({X[i]})({W1[i, j]})" for i in range(2))
        vals = {"0_0": "1.0", "0_1": "2.0"}
        for jj in range(j):
            vals[f"1_{jj}"] = f"{a1[jj]:.2f}"
        vals[f"1_{j}"] = f"{a1[j]:.2f}"
        frames.append(Frame(_svg({"values": vals, "active": [f"1_{j}"]}),
                            caption(f"**الخلية المخفية {j + 1}**: `z = {terms} + {B1[j]} = {z1[j]:+.2f}` ثم `ReLU → a = {a1[j]:.2f}`" + (" (**صُفِّرت**: z سالب)" if z1[j] < 0 else "")),
                            action=f"Hidden unit {j + 1}", equation=f"z1[{j}] = {z1[j]:+.3f};  a1[{j}] = max(0, z) = {a1[j]:.3f}", highlight=1))
    vals = {"0_0": "1.0", "0_1": "2.0", **{f"1_{j}": f"{a1[j]:.2f}" for j in range(3)}, "2_0": f"{p[0]:.2f}"}
    terms2 = " + ".join(f"({a1[j]:.2f})({W2[j, 0]})" for j in range(3))
    frames.append(Frame(_svg({"values": vals, "active": ["2_0"]}), caption(f"**الإخراج**: `z = {terms2} + {B2[0]} = {z2[0]:+.2f}`، ثم `Sigmoid → p = {p[0]:.3f}`، القرار بعتبة 0.5: **{int(p[0] >= 0.5)}**."),
                        action="Output", equation=f"z2 = {z2[0]:+.3f};  p = sigmoid(z2) = {p[0]:.3f}", values=[("p", "", f"{p[0]:.3f}"), ("class", "", str(int(p[0] >= 0.5)))], highlight=2))
    animation_player("fwd_anim", frames, title_ar="التمرير الأمامي خلية بخلية", stages=["Input", "Hidden", "Output"], interval_ms=2200)
    worked_steps([("z₁ = xW₁ + b₁", rf"({z1[0]:+.2f},\ {z1[1]:+.2f},\ {z1[2]:+.2f})"), ("a₁ = ReLU(z₁)", rf"({a1[0]:.2f},\ {a1[1]:.2f},\ {a1[2]:.2f})"),
                  ("z₂ = a₁W₂ + b₂", rf"{z2[0]:+.3f}"), ("p = σ(z₂)", rf"{p[0]:.3f} \Rightarrow \hat y = {int(p[0] >= 0.5)}")], title_ar="الحساب بالمصفوفات")
    quiz("fwd.walk", [
        Q("الخلية المخفية 2 أعطت z سالبًا. مخرجها بعد ReLU…", ["z نفسه", "0", "1"], 1, "التصفير."),
        Q("مساهمة الخلية المخفية 2 في الإخراج بعد التصفير…", ["كبيرة", "صفر", "سالبة"], 1, "0 × وزنها."),
        Q("لو استبدلنا ReLU بـ tanh، هل تبقى الخلية 2 «صامتة»؟", ["نعم", "لا: tanh(z سالب) قيمة سالبة غير صفرية", "لا تُحسب"], 1, "لا تصفير في tanh."),
    ])
    takeaway("كل خلية: ضرب نقطي + انحياز + تنشيط. تصفير ReLU يقطع مساهمة الخلية بالكامل عن الطبقة التالية.")
    lesson_footer(LESSON, ["أرقام حقيقية لكل z وa.", "الحساب بالمصفوفات يطابق الحساب خلية بخلية.", "التنشيط يحدد ما يمر."])
