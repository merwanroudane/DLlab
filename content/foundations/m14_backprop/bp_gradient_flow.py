import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway
from components.diagram import diagram, svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.backprop.backpropagation.gradient_flow",
    title_ar="تدفق التدرج: δ والمصفوفات عبر الطبقات",
    title_en="Gradient Flow: δ and the Matrix Rules Across Layers",
    module="foundations.backprop",
    parent="foundations.backprop.backpropagation",
    order=2,
    prerequisites=["foundations.backprop.backpropagation"],
    objectives_ar=["تعريف δ لكل طبقة وقواعد الانتقال بين الطبقات بصيغة المصفوفات.", "التحقق من أشكال كل تدرج بأنه يطابق شكل معلمته."],
    terms=["gradient", "matrix_multiplication", "transpose"],
    difficulty="intermediate",
    summary_ar="δ^(L) من الخسارة؛ δ^(ℓ−1) = (δ^(ℓ) W^(ℓ)ᵀ) ⊙ f'(z^(ℓ−1))؛ ∂L/∂W^(ℓ) = a^(ℓ−1)ᵀ δ^(ℓ)؛ ∂L/∂b = Σ δ.",
)


def _flow_svg() -> str:
    s = '<svg viewBox="0 0 660 200" width="100%" style="max-width:660px">' + svg_defs()
    xs = [30, 200, 370, 540]
    labels = ["a⁽⁰⁾ = X", "z⁽¹⁾ → a⁽¹⁾", "z⁽²⁾ → a⁽²⁾", "L"]
    for x, lbl in zip(xs, labels):
        s += svg_box(x, 40, 110, 46, lbl, "#E6F1FB", stroke="#2F6FB5", font=13, bold=True)
    for i in range(3):
        s += svg_arrow(xs[i] + 110, 63, xs[i + 1] - 3, 63)
    s += svg_text(330, 25, "forward (black)", size=12, color="#2F6FB5")
    for i in range(3, 0, -1):
        s += f'<path d="M{xs[i] + 10},95 C{xs[i] - 40},135 {xs[i - 1] + 150},135 {xs[i - 1] + 110},95" fill="none" stroke="#C8473A" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#arrowhead)"/>'
    s += svg_text(600, 130, "δ⁽²⁾ = ∂L/∂z⁽²⁾", size=11, color="#C8473A") + svg_text(300, 150, "δ⁽¹⁾ = (δ⁽²⁾ W⁽²⁾ᵀ) ⊙ f′(z⁽¹⁾)", size=11, color="#C8473A")
    s += svg_text(330, 185, "∂L/∂W⁽ℓ⁾ = a⁽ℓ−1⁾ᵀ δ⁽ℓ⁾      ∂L/∂b⁽ℓ⁾ = Σ_rows δ⁽ℓ⁾", size=12, mono=True)
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    diagram("تدفق التدرج عبر طبقتين", _flow_svg(), what_ar="الأسهم السوداء: التمرير الأمامي يحفظ a وz. الأسهم الحمراء: δ ينتقل من الخسارة إلى الخلف، وعند كل طبقة يُستخرج منه تدرج W وb.",
            how_ar="ابدأ من اليمين: δ⁽²⁾ من مشتقة الخسارة. اضربه في W⁽²⁾ᵀ ليعود إلى الطبقة السابقة، ثم في f′ ليمر عبر التنشيط. عند كل طبقة: a_prevᵀ δ يعطي تدرج الأوزان.",
            takeaway_ar="ثلاث عمليات فقط تتكرر: ضرب في Wᵀ، ضرب عنصري في f′، وضرب a_prevᵀ δ.", title_en="Gradient flow")
    definition("**δ⁽ℓ⁾** `delta`: تدرج الخسارة بالنسبة لما قبل التنشيط في الطبقة $\\ell$، بالشكل `(n, units_ℓ)`: رقم لكل ملاحظة ولكل خلية = «كم يجب أن يتغير z هذه الخلية لهذه الملاحظة».")
    equation(r"\delta^{(L)} = \frac{\partial L}{\partial z^{(L)}} \quad(\text{مثلًا } (p - y)/n), \qquad \delta^{(\ell-1)} = \big(\delta^{(\ell)} W^{(\ell)\mathsf T}\big) \odot f'\big(z^{(\ell-1)}\big)",
             [(r"\delta^{(\ell)} W^{(\ell)\mathsf T}", "`(n, out)@(out, in)` → `(n, in)`: توزيع خطأ كل خلية على مدخلاتها بأوزانها."), (r"\odot f'(z^{(\ell-1)})", "هادامارد بمشتقة التنشيط: مقدار ما تمرره كل خلية سابقة (0 لـ ReLU الميتة، ≤0.25 لـ Sigmoid).")],
             meaning_ar="الخطأ يعود عبر الأوزان (منقولة) ثم يُرشَّح بمشتقة التنشيط.",
             example_ar="طبقة أخيرة `(n, 1)` بـ Sigmoid+BCE: $\\delta^{(2)} = (p - y)/n$؛ $W^{(2)}$ بالشكل `(16, 1)` ⇒ $\\delta^{(2)} W^{(2)\\mathsf T}$ بالشكل `(n, 16)`.",
             dl_link_ar="الضرب في $W^{\\mathsf T}$ هو سبب ظهور `W.T` في كل تنفيذ يدوي، وسبب أن الأوزان الكبيرة تضخّم التدرج (انفجار) والصغيرة تخمده (تلاشٍ).", title_ar="انتقال δ إلى الخلف")
    equation(r"\frac{\partial L}{\partial W^{(\ell)}} = a^{(\ell-1)\mathsf T}\,\delta^{(\ell)}, \qquad \frac{\partial L}{\partial b^{(\ell)}} = \sum_{i=1}^{n}\delta^{(\ell)}_{i,:}",
             [(r"a^{(\ell-1)\mathsf T}\delta^{(\ell)}", "`(in, n)@(n, out)` → `(in, out)` = شكل $W$ بالضبط: مجموع على الملاحظات لحاصل (مدخل × خطأ)."), (r"\sum_i \delta", "الانحياز يتلقى مجموع الأخطاء عبر الملاحظات: شكل `(out,)`.")],
             meaning_ar="تدرج وزن الرابط (j→k) = مجموع (تنشيط j × خطأ k) عبر الدفعة. وزن يربط مدخلًا نشطًا بخلية مخطئة يُعدَّل أكثر.",
             example_ar="`a_prev (32, 16)`, `δ (32, 8)` ⇒ `gW (16, 8)`، `gb (8,)`.",
             dl_link_ar="فحص الأشكال: كل تدرج يجب أن يطابق شكل معلمته. `assert gW.shape == W.shape` أول اختبار لأي تنفيذ.", title_ar="تدرجات المعلمات")
    intuition("«الخطأ يسافر إلى الخلف عبر الطرق التي جاء منها الإشارة إلى الأمام»: نفس الأوزان، اتجاه معكوس (منقولة)، مع بوابات (مشتقات التنشيط) تقرر كم يمر.")
    common_mistake("نسيان `⊙ f'(z)` عند العودة عبر طبقة مخفية: التدرج يبدو صحيح الشكل لكنه خاطئ القيمة. التحقق العددي (الصفحة الفرعية 4) يكشفه فورًا.")
    quiz("bp.flow", [
        Q("`δ (64, 10)`, `W (32, 10)`: شكل `δ @ W.T`…", ["(64, 32)", "(32, 64)", "(10, 32)"], 0, "يعود إلى مدخلات الطبقة.", kind="shape"),
        Q("`a_prev (64, 32)`, `δ (64, 10)`: شكل تدرج W…", ["(64, 64)", "(32, 10)", "(10, 32)"], 1, "يطابق W.", kind="shape"),
        Q("ما الذي يرشّح δ عند المرور عبر التنشيط؟", ["W", "مشتقة التنشيط f'(z)", "الانحياز"], 1, "هادامارد."),
    ])
    takeaway("δ ينتقل بـ Wᵀ ويُرشَّح بـ f′؛ gW = a_prevᵀ δ، gb = Σδ. الأشكال تتطابق مع المعلمات دائمًا.")
    lesson_footer(LESSON, ["ثلاث عمليات متكررة.", "assert gW.shape == W.shape.", "Wᵀ يفسر التلاشي والانفجار."])
