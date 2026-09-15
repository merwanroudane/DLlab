import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway
from components.comparison import compare_table
from components.diagram import diagram, svg_text
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson


LESSON = Lesson(
    id="foundations.architecture.layers",
    title_ar="طبقات الإدخال والمخفية والإخراج، والروابط",
    title_en="Input, Hidden & Output Layers; Connections",
    module="foundations.architecture",
    order=1,
    prerequisites=["foundations.neuron.neuron_perceptron"],
    objectives_ar=["تسمية الطبقات الثلاث ووظيفة كل منها.", "قراءة رسم الشبكة: كل رابط وزن، كل عقدة مخفية خلية بانحياز.", "كتابة الشبكة كسلسلة معادلات مصفوفية."],
    terms=["matrix", "weight", "bias"],
    labs=["labs.network_builder"],
    difficulty="beginner",
    summary_ar="إدخال (بلا معلمات) → مخفية (خلايا بتنشيط) → إخراج (تنشيط حسب المهمة). كل رابط وزن.",
)


def _net_svg(sizes=(3, 4, 2)) -> str:
    W, H = 620, 260
    cols = [60, 310, 560]
    s = f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px">'
    colors = ["#E6F1FB", "#EFE9FA", "#DDF5EA"]; strokes = ["#2F6FB5", "#7C5CBF", "#2E8B57"]
    pos = []
    for li, n in enumerate(sizes):
        ys = [H / 2 + (i - (n - 1) / 2) * 50 for i in range(n)]
        pos.append([(cols[li], y) for y in ys])
    for li in range(len(sizes) - 1):
        for (x1, y1) in pos[li]:
            for (x2, y2) in pos[li + 1]:
                s += f'<line x1="{x1 + 18}" y1="{y1}" x2="{x2 - 18}" y2="{y2}" stroke="#C9C1B3" stroke-width="1"/>'
    for li, layer in enumerate(pos):
        for (x, y) in layer:
            s += f'<circle cx="{x}" cy="{y}" r="18" fill="{colors[li]}" stroke="{strokes[li]}" stroke-width="2"/>'
    s += svg_text(60, 30, "Input (3)", size=13, bold=True, color="#2F6FB5") + svg_text(310, 30, "Hidden (4) + ReLU", size=13, bold=True, color="#7C5CBF") + svg_text(560, 30, "Output (2)", size=13, bold=True, color="#2E8B57")
    s += svg_text(185, 245, "W₁: (3, 4), b₁: (4,)", size=12, mono=True) + svg_text(435, 245, "W₂: (4, 2), b₂: (2,)", size=12, mono=True)
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("ثلاث طبقات، دوران مختلفان", "Three layers")
    diagram("شبكة كثيفة 3 → 4 → 2", _net_svg(), what_ar="ثلاثة أعمدة من الدوائر: الإدخال (3 خصائص)، طبقة مخفية (4 خلايا)، الإخراج (2 وحدة). كل خط رابط له وزن.",
            how_ar="عدّ الخطوط بين عمودين = عدد الأوزان بينهما (3×4 = 12، ثم 4×2 = 8). كل دائرة مخفية أو إخراج لها انحياز إضافي. دوائر الإدخال ليست خلايا: مجرد قيم.",
            takeaway_ar="الروابط هي المعلمات. الطبقات المخفية والإخراج تحسب؛ الإدخال يعرض فقط.", title_en="A dense network",
            legend=[("#E6F1FB", "إدخال"), ("#EFE9FA", "مخفية"), ("#DDF5EA", "إخراج")])
    compare_table(["الطبقة", "English", "ماذا تفعل", "لها معلمات؟", "التنشيط"],
                  [("الإدخال", "Input layer", "تعرض الخصائص كما هي (بعد التحضير)", "لا", "لا"),
                   ("المخفية", "Hidden layer(s)", "تحسب تمثيلات وسيطة: مجموع موزون + تنشيط لكل خلية", "نعم: W وb", "ReLU عادةً"),
                   ("الإخراج", "Output layer", "تنتج التنبؤ بالشكل الذي تحتاجه المهمة", "نعم: W وb", "حسب المهمة: خطي / Sigmoid / Softmax")],
                  ["rtl", "ltr", "rtl", "rtl", "rtl"])
    definition("**الطبقة الكثيفة** `Dense` / `Fully connected`: كل خلية فيها متصلة بكل مخرجات الطبقة السابقة. **الرابط** `Connection` وزن واحد. طبقة الإدخال لا تحسب شيئًا؛ تحدد فقط عدد المدخلات.")
    equation(r"\mathbf{a}^{(1)} = f\big(\mathbf{x}W_1 + \mathbf{b}_1\big), \qquad \hat{\mathbf{y}} = g\big(\mathbf{a}^{(1)}W_2 + \mathbf{b}_2\big)",
             [(r"\mathbf{x}", "متجه المدخل `(3,)` (أو دفعة `(n, 3)`)."), ("W_1", "أوزان الطبقة المخفية `(3, 4)`: عمود لكل خلية."), (r"\mathbf{b}_1", "انحياز لكل خلية `(4,)`."), ("f", "تنشيط المخفية (ReLU)."), ("W_2, \\mathbf{b}_2", "الإخراج `(4, 2)` و`(2,)`."), ("g", "تنشيط الإخراج حسب المهمة.")],
             meaning_ar="الشبكة سلسلة من (ضرب مصفوفات + انحياز + تنشيط). كل سطر طبقة.", example_ar="دفعة `(32, 3)` ← `(32, 4)` ← `(32, 2)`: المحور 0 (الدفعة) لا يتغير أبدًا.",
             dl_link_ar="هذه معادلات `Sequential([Dense(4, 'relu'), Dense(2, 'softmax')])` بالضبط.", title_ar="الشبكة كمعادلات")
    intuition("كل خلية مخفية «كاشف نمط» صغير: تتعلم تركيبة من المدخلات تستحق الانتباه. الإخراج يجمع آراء الكواشف. مزيد من الكواشف (عرض) أو طبقات من الكواشف فوق كواشف (عمق) = قدرة أكبر.")
    common_mistake("عدّ «3 طبقات» في الرسم أعلاه. الاصطلاح: نعدّ الطبقات ذات المعلمات — شبكة **بطبقتين** (مخفية + إخراج). Keras لا تُنشئ كائنًا لطبقة الإدخال؛ `input_shape` صفة لأول طبقة كثيفة.")
    quiz("arch.layers", [
        Q("شبكة 8 → 16 → 16 → 1: كم طبقة ذات معلمات؟", ["4", "3", "2"], 1, "مخفيتان + إخراج."),
        Q("عدد الأوزان بين طبقتين من 5 و7 وحدات…", ["12", "35", "7"], 1, "5×7.", kind="shape"),
        Q("طبقة الإدخال…", ["لها أوزان", "لا معلمات لها", "لها انحياز فقط"], 1, "تعرض القيم."),
        Q("تنشيط طبقة الإخراج يحدده…", ["عدد الطبقات المخفية", "نوع المهمة", "معدل التعلم"], 1, "خطي/Sigmoid/Softmax."),
    ])
    takeaway("إدخال يعرض، مخفية تحسب بتنشيط، إخراج ينتج بتنشيط المهمة. كل رابط وزن، كل خلية انحياز، وكل طبقة سطر مصفوفي.")
    lesson_footer(LESSON, ["الطبقات ذات المعلمات هي ما نعدّه.", "W (in, out) وb (out,) لكل طبقة.", "الدفعة تمر دون تغيير محورها 0."])
