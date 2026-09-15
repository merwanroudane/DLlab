import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.diagram import diagram, svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation, worked_steps
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go

LESSON = Lesson(
    id="foundations.calculus.chain_rule",
    title_ar="قاعدة السلسلة والسلسلة الحسابية",
    title_en="The Chain Rule & Computational Chains",
    module="foundations.calculus",
    order=4,
    prerequisites=["foundations.calculus.partial_gradient"],
    objectives_ar=[
        "اشتقاق تركيب دوال $f(g(x))$ بضرب المشتقات المحلية.",
        "رسم سلسلة حسابية (مدخل ← خطي ← تنشيط ← خسارة) واشتقاقها من النهاية إلى البداية.",
        "إدراك أن الانتشار الخلفي هو قاعدة السلسلة مطبقة على رسم حسابي.",
    ],
    terms=["gradient", "loss", "weight"],
    labs=["labs.chain_rule_lab"],
    related=["foundations.calculus.partial_gradient"],
    difficulty="intermediate",
    summary_ar="مشتقة التركيب = حاصل ضرب المشتقات المحلية؛ الانتشار الخلفي يطبقها على السلسلة من الخسارة إلى الوزن.",
)

CODE = '''import numpy as np

x, y, w, b = 2.0, 1.0, 0.5, -0.3
sigmoid = lambda z: 1 / (1 + np.exp(-z))

# التمرير الأمامي: سلسلة من ثلاث خطوات
z = w * x + b                    # خطي
a = sigmoid(z)                   # تنشيط
L = (a - y) ** 2                 # خسارة
print(f"forward: z={z:.4f} a={a:.4f} L={L:.4f}")

# المشتقات المحلية (كل حلقة في السلسلة)
dL_da = 2 * (a - y)              # ∂L/∂a
da_dz = a * (1 - a)              # ∂a/∂z  (مشتقة sigmoid)
dz_dw = x                        # ∂z/∂w
dz_db = 1.0                      # ∂z/∂b

# قاعدة السلسلة: اضرب على طول المسار من L إلى w
dL_dz = dL_da * da_dz
dL_dw = dL_dz * dz_dw
dL_db = dL_dz * dz_db
print(f"backward: dL/da={dL_da:.4f} da/dz={da_dz:.4f} dL/dz={dL_dz:.4f}")
print(f"          dL/dw={dL_dw:.4f} dL/db={dL_db:.4f}")

# تحقق عددي
def Lw(w_): return (sigmoid(w_ * x + b) - y) ** 2
h = 1e-5
print(f"numeric dL/dw={(Lw(w + h) - Lw(w - h)) / (2*h):.4f}")'''


def _chain_svg(active: int | None = None) -> str:
    s = '<svg viewBox="0 0 640 170" width="100%" style="max-width:640px">' + svg_defs()
    boxes = [("x, w, b", "#E6F1FB", "#2F6FB5"), ("z = wx + b", "#EFE9FA", "#7C5CBF"), ("a = σ(z)", "#FFF1DC", "#C77A1A"), ("L = (a − y)²", "#FBE6E2", "#C8473A")]
    xs = [20, 180, 340, 500]
    for i, ((label, fill, stroke), x) in enumerate(zip(boxes, xs)):
        s += svg_box(x, 30, 130, 50, label, fill, stroke=("#1F7A78" if i == active else stroke), font=13, bold=True)
    for i in range(3):
        s += svg_arrow(xs[i] + 130, 55, xs[i + 1] - 3, 55)
    labels = ["∂z/∂w = x", "∂a/∂z = a(1−a)", "∂L/∂a = 2(a−y)"]
    for i, lab in enumerate(labels):
        s += svg_text(xs[i] + 130 + 15, 110, lab, size=12, color="#6B675F", anchor="start")
        s += f'<path d="M{xs[i + 1] + 65},95 C{xs[i + 1] + 40},120 {xs[i] + 100},120 {xs[i] + 80},95" fill="none" stroke="#C8473A" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arrowhead)"/>'
    s += svg_text(320, 150, "forward → (black)     backward ← (red dashed): dL/dw = dL/da · da/dz · dz/dw", size=12, color="#2B2A28")
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("القاعدة", "The rule")
    equation(r"\frac{d}{dx} f\big(g(x)\big) = f'\big(g(x)\big)\cdot g'(x) \qquad\text{أو}\qquad \frac{dL}{dx} = \frac{dL}{du}\cdot\frac{du}{dx}",
             [("f(g(x))", "دالة داخل دالة: أولًا $g$ ثم $f$."), ("f'(g(x))", "مشتقة الخارجية **عند قيمة الداخلية**."), ("g'(x)", "مشتقة الداخلية."), (r"\frac{dL}{du}\cdot\frac{du}{dx}", "بصيغة لايبنتز: «الوسيط» $u$ يُختصر كأنه كسر.")],
             meaning_ar="مشتقة التركيب = حاصل ضرب المشتقات المحلية على طول السلسلة.",
             example_ar="$L = (3x + 1)^2$: خارجية $u^2 \\to 2u$، داخلية $3x+1 \\to 3$. $\\frac{dL}{dx} = 2(3x+1)\\cdot 3$؛ عند $x = 1$: $24$.",
             dl_link_ar="الخسارة تعتمد على المخرج الذي يعتمد على التنشيط الذي يعتمد على المجموع الموزون الذي يعتمد على الوزن. أربع حلقات ← ثلاثة ضربات. هذا هو الانتشار الخلفي.", title_ar="قاعدة السلسلة")
    intuition("سلسلة تروس: لو دار الترس الأول درجة واحدة، فكم يدور الأخير؟ اضرب نسب التروس المتتالية. كل مشتقة محلية نسبة ترس.")
    diagram("السلسلة الحسابية لخلية واحدة", _chain_svg(), what_ar="أربعة صناديق: المدخلات والمعلمات، المجموع الخطي، التنشيط، الخسارة. الأسهم السوداء تمرير أمامي؛ المتقطعة الحمراء عودة المشتقات.",
            how_ar="للحصول على ∂L/∂w اتبع المسار الأحمر من L إلى w واضرب المشتقات المحلية المكتوبة تحت الأسهم.",
            takeaway_ar="كل صندوق يعرف مشتقته المحلية فقط؛ قاعدة السلسلة تربطها. لا يحتاج أي صندوق معرفة ما وراء جاره.", title_en="Computational chain")
    h3("تحريك: التمرير الأمامي ثم الخلفي", "Animation")
    x, y, w, b = 2.0, 1.0, 0.5, -0.3
    z = w * x + b; a = 1 / (1 + 2.718281828 ** (-z)); L = (a - y) ** 2
    dL_da = 2 * (a - y); da_dz = a * (1 - a); dL_dz = dL_da * da_dz; dL_dw = dL_dz * x
    frames = [
        Frame(_chain_svg(0), caption(f"**أمامي 1/3** — المدخلات: `x = {x}`, `w = {w}`, `b = {b}`, الهدف `y = {y}`."), action="Forward", highlight=0),
        Frame(_chain_svg(1), caption(f"**أمامي 2/3** — المجموع الخطي `z = w·x + b = {z:.3f}`."), action="Forward", equation=f"z = {w}×{x} + ({b}) = {z:.3f}", highlight=1),
        Frame(_chain_svg(2), caption(f"**أمامي 3/3** — التنشيط `a = σ(z) = {a:.4f}`، ثم الخسارة `L = (a − y)² = {L:.4f}`."), action="Forward", equation=f"a = σ({z:.3f}) = {a:.4f};  L = ({a:.4f} − 1)² = {L:.4f}", highlight=2),
        Frame(_chain_svg(3), caption(f"**خلفي 1/3** — نبدأ من الخسارة: `∂L/∂a = 2(a − y) = {dL_da:.4f}`."), action="Backward", equation=f"dL/da = 2({a:.4f} − 1) = {dL_da:.4f}", highlight=3),
        Frame(_chain_svg(2), caption(f"**خلفي 2/3** — نمر عبر التنشيط: `∂a/∂z = a(1 − a) = {da_dz:.4f}`؛ نضرب: `∂L/∂z = {dL_dz:.4f}`."), action="Backward", equation=f"dL/dz = {dL_da:.4f} × {da_dz:.4f} = {dL_dz:.4f}", highlight=2),
        Frame(_chain_svg(1), caption(f"**خلفي 3/3** — نمر عبر الخطي: `∂z/∂w = x = {x}`؛ النتيجة `∂L/∂w = {dL_dw:.4f}`. إشارتها سالبة: زيادة w تقلل الخسارة."), action="Backward", equation=f"dL/dw = {dL_dz:.4f} × {x} = {dL_dw:.4f}", values=[("dL/dw", "", f"{dL_dw:.4f}"), ("dL/db", "", f"{dL_dz:.4f}")], highlight=1),
    ]
    animation_player("chain_anim", frames, title_ar="أمامي ثم خلفي عبر السلسلة", stages=["Forward", "Backward"], interval_ms=1600)
    worked_steps([("المشتقات المحلية", r"\tfrac{\partial L}{\partial a} = 2(a-y),\quad \tfrac{\partial a}{\partial z} = a(1-a),\quad \tfrac{\partial z}{\partial w} = x"),
                  ("الضرب على طول المسار", r"\tfrac{\partial L}{\partial w} = 2(a-y)\cdot a(1-a)\cdot x"),
                  ("بالأرقام", rf"= ({dL_da:.4f})({da_dz:.4f})({x}) = {dL_dw:.4f}")], title_ar="اشتقاق ∂L/∂w")
    code_lab(CodeLab(
        key="calc_chain", title_ar="أمامي وخلفي بأيدينا + تحقق عددي", code=CODE,
        before=Before(goal_ar="تنفيذ التمرير الأمامي لخلية واحدة، ثم حساب ∂L/∂w و∂L/∂b بقاعدة السلسلة، والتحقق عدديًا.", stage_ar="تفاضل ← الانتشار الخلفي.",
                      inputs_ar="أعداد: `x, y, w, b`.", expected_ar="قيم أمامية، مشتقات محلية، تدرجان، وتحقق عددي مطابق.",
                      math_ar="$\\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial a}\\frac{\\partial a}{\\partial z}\\frac{\\partial z}{\\partial w}$."),
        explain=[("6-10", "ثلاث خطوات أمامية؛ نحفظ القيم الوسيطة `z, a` لأن الخلفي يحتاجها."),
                 ("12-16", "مشتقة محلية لكل صندوق — كل واحدة من جدول الدرس السابق."),
                 ("18-23", "الضرب من النهاية: `dL/dz` مشترك بين `w` و`b` (يُحسب مرة واحدة ويُعاد استخدامه — هذا سر كفاءة الانتشار الخلفي)."),
                 ("25-28", "الفروق المنتهية تؤكد النتيجة.")],
        run=run_printed(CODE),
        after_ar="- `dL/dz` يُحسب مرة ويُستخدم لكل معلمات الطبقة. في شبكة كبيرة توفر هذه المشاركة ملايين العمليات.\n- التحقق العددي = 0.0000 فرق: القاعدة صحيحة.",
    ))
    why("لماذا الاتجاه من الخسارة إلى الوزن (خلفي) لا العكس؟ لأن $\\partial L/\\partial z$ الوسيط يخدم كل المعلمات قبله؛ الحساب من النهاية يعيد استخدامه، بينما من البداية يكرره لكل معلمة.")
    common_mistake("نسيان ضرب المشتقة الخارجية عند قيمة الداخلية: $\\frac{d}{dx}\\sigma(3x) = \\sigma'(3x)\\cdot 3$ وليس $\\sigma'(x)\\cdot 3$.")
    st.button("افتح معمل قاعدة السلسلة", icon=":material/science:", on_click=go, args=("labs.chain_rule_lab",), key="chain_lab_btn")
    quiz("calc.chain", [
        Q("$\\frac{d}{dx}(2x + 1)^3$ عند $x = 0$…", ["3", "6", "2"], 1, "3(1)² × 2.", kind="equation"),
        Q("في سلسلة L ← a ← z ← w، $\\partial L/\\partial w$ يساوي…", ["مجموع الجزئيات", "حاصل ضرب الجزئيات المحلية", "الجزئية الأخيرة فقط"], 1, "قاعدة السلسلة."),
        Q("لماذا نحسب من الخسارة إلى الخلف؟", ["لأن الرياضيات تفرضه", "لإعادة استخدام المشتقات الوسيطة لكل المعلمات", "لأنه أسهل قراءة"], 1, "كفاءة."),
        Q("إذا كانت مشتقة التنشيط 0.01 في كل طبقة من 5 طبقات، فحاصل ضربها…", ["0.05", "1e-10", "0.01"], 1, "تلاشي التدرج.", kind="equation"),
    ])
    takeaway("مشتقة التركيب = ضرب المحليات. الانتشار الخلفي = قاعدة السلسلة على السلسلة من L إلى w مع إعادة استخدام الوسائط. حاصل ضرب أعداد صغيرة يتلاشى.")
    lesson_footer(LESSON, ["f(g(x))' = f'(g)·g'.", "احفظ الوسائط أمامًا، اضرب خلفًا.", "dL/dz مشترك؛ ضرب مشتقات صغيرة = تلاشٍ."])
