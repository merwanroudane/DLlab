import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import common_mistake, debugging_note, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.diagram import svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from labs.fw import torch

LESSON = Lesson(
    id="foundations.frameworks.pytorch.autograd",
    title_ar="Autograd: requires_grad و.grad والرسم الحسابي وbackward — محرك مرئي",
    title_en="Autograd: requires_grad, .grad, the Computation Graph & backward() — Visualizer",
    module="foundations.frameworks",
    parent="foundations.frameworks.pytorch",
    order=21,
    prerequisites=["foundations.frameworks.pytorch.tensors", "foundations.frameworks.tensorflow.gradient_tape", "foundations.calculus.chain_rule"],
    objectives_ar=["فهم كيف يبني PyTorch الرسم الحسابي ديناميكيًا أثناء التمرير الأمامي (grad_fn).", "ما يحدث عند loss.backward(): المرور على الرسم عكسيًا وتخزين التدرجات في .grad لكل ورقة بـ requires_grad.", "لماذا تتراكم .grad ولماذا يجب تصفيرها في حلقة التدريب."],
    terms=["gradient", "chain_rule", "derivative", "backpropagation"],
    labs=["labs.gradient_tape_lab"],
    difficulty="intermediate",
    summary_ar="كل عملية على موتر بـ requires_grad تسجّل grad_fn وتبني رسمًا. loss.backward() يمشي على الرسم إلى الخلف بقاعدة السلسلة ويجمع (+=) النتيجة في .grad للأوراق. لذلك zero_grad قبل كل backward.",
)

CODE = '''import torch
# y_hat = w·x + b ;  L = (y_hat − y)²     عند x=2, y=1, w=3, b=0.5
x = torch.tensor(2.0); y = torch.tensor(1.0)
w = torch.tensor(3.0, requires_grad=True)          # ورقة مراقَبة
b = torch.tensor(0.5, requires_grad=True)          # ورقة مراقَبة
y_hat = w * x + b                                  # الرسم يُبنى الآن: MulBackward -> AddBackward
loss = (y_hat - y) ** 2                            # SubBackward -> PowBackward
print("y_hat =", y_hat.item(), "| loss =", loss.item())
print("grad_fn chain:", type(loss.grad_fn).__name__, "<-", type(y_hat.grad_fn).__name__, "| x.grad_fn =", x.grad_fn, "(leaf, no grad)")
print("before backward: w.grad =", w.grad, "b.grad =", b.grad)
loss.backward()                                    # قاعدة السلسلة إلى الخلف؛ تملأ .grad للأوراق
print("after  backward: w.grad =", w.grad.item(), "b.grad =", b.grad.item(), "| hand: dL/dw = 2(ŷ−y)·x =", 2 * (6.5 - 1) * 2, ", dL/db = 2(ŷ−y) =", 2 * (6.5 - 1))
print("y_hat.grad =", y_hat.grad, "(intermediate: not kept)")

# التراكم: backward ثانية تجمع على .grad
y_hat = w * x + b; loss = (y_hat - y) ** 2
loss.backward()
print("second backward WITHOUT zeroing: w.grad =", w.grad.item(), "(= 22 + 22: accumulated!)")
w.grad.zero_(); b.grad.zero_()                     # ما يفعله optimizer.zero_grad()
y_hat = w * x + b; loss = (y_hat - y) ** 2; loss.backward()
print("after zero_ then backward       : w.grad =", w.grad.item())

# خطوة تحديث يدوية خارج الرسم
with torch.no_grad():                              # لا تسجّل التحديث نفسه في الرسم
    w -= 0.01 * w.grad; b -= 0.01 * b.grad
print("after manual SGD step: w =", round(w.item(), 3), "b =", round(b.item(), 3), "| requires_grad still", w.requires_grad)
# detach / no_grad للاستدلال
with torch.no_grad():
    pred = w * x + b
print("under no_grad: pred.requires_grad =", pred.requires_grad, "| detach():", (w * x).detach().requires_grad)'''


def _graph_svg(stage: int, vals: dict) -> str:
    """Computation graph: leaves w, x, b, y → mul → add → sub → pow → loss.
    stage 0..4 forward highlight; 5..8 backward highlight (red)."""
    s = '<svg viewBox="0 0 700 260" width="100%" style="max-width:700px">' + svg_defs()
    nodes = {"w": (30, 40), "x": (30, 110), "b": (30, 180), "mul": (170, 75), "add": (310, 110), "y": (310, 200), "sub": (450, 150), "pow": (570, 150)}
    fwd_order = ["leaves", "mul", "add", "sub", "pow"]
    bwd = {5: "pow", 6: "sub", 7: "add", 8: "mul"}
    labels = {"w": f"w = {vals['w']:.2f}", "x": f"x = {vals['x']:.2f}", "b": f"b = {vals['b']:.2f}", "y": f"y = {vals['y']:.2f}", "mul": f"w·x = {vals['wx']:.2f}", "add": f"ŷ = {vals['yhat']:.2f}", "sub": f"ŷ−y = {vals['diff']:.2f}", "pow": f"L = {vals['loss']:.2f}"}
    for k, (x, y) in nodes.items():
        leaf = k in ("w", "x", "b", "y")
        fill = "#E6F1FB" if leaf else "#F1EFEA"
        stroke = "#B9B2A6"
        if stage >= 1 and k == fwd_order[min(stage, 4)] and stage <= 4:
            fill, stroke = "#E3F3F0", "#1F7A78"
        if stage == 0 and leaf:
            fill, stroke = "#E3F3F0", "#1F7A78"
        if stage >= 5 and bwd.get(stage) == k:
            fill, stroke = "#FBE6E2", "#C8473A"
        if stage >= 9 and k in ("w", "b"):
            fill, stroke = "#FBE6E2", "#C8473A"
        s += svg_box(x, y, 110, 40, labels[k], fill, stroke=stroke, font=12, bold=not leaf)
    edges = [("w", "mul"), ("x", "mul"), ("mul", "add"), ("b", "add"), ("add", "sub"), ("y", "sub"), ("sub", "pow")]
    for a, c in edges:
        (ax, ay), (cx, cy) = nodes[a], nodes[c]
        s += svg_arrow(ax + 112, ay + 20, cx - 2, cy + 20)
    if stage >= 5:
        grads = {"pow": f"∂L/∂L = 1", "sub": f"∂L/∂(ŷ−y) = {vals['g_diff']:.2f}", "add": f"∂L/∂ŷ = {vals['g_diff']:.2f}", "mul": f"∂L/∂(w·x) = {vals['g_diff']:.2f}"}
        for k in ["pow", "sub", "add", "mul"][: stage - 4]:
            x, y = nodes[k]
            s += svg_text(x + 55, y + 58, grads[k], size=10, color="#C8473A", mono=True)
    if stage >= 9:
        s += svg_text(85, 30, f"w.grad = {vals['gw']:.2f}", size=11, color="#C8473A", bold=True, mono=True) + svg_text(85, 240, f"b.grad = {vals['gb']:.2f}", size=11, color="#C8473A", bold=True, mono=True)
        s += svg_text(85, 100, "x.grad = None", size=10, color="#6B675F", mono=True)
    s += svg_text(350, 20, "forward builds the graph left→right; backward walks it right→left", size=11, color="#6B675F")
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    definition("**Autograd**: محرك الاشتقاق التلقائي في PyTorch. كل موتر بـ `requires_grad=True` (ورقة) وكل عملية عليه تنتج موترًا يحمل `grad_fn` — هكذا يُبنى **الرسم الحسابي ديناميكيًا** أثناء التمرير الأمامي. `loss.backward()` يمشي على الرسم من الخسارة إلى الأوراق بقاعدة السلسلة و**يجمع** النتيجة في `.grad` لكل ورقة.")
    why("في TensorFlow كتبت `with GradientTape()` صراحةً لتقول «سجّل الآن». في PyTorch التسجيل **دائم** لكل موتر بـ `requires_grad`: لا سياق، فقط `backward()`. الثمن: التدرجات **تُجمَع** في `.grad` ولا تُستبدل — لذلك يوجد `zero_grad()` في كل حلقة، وهذا الدرس يريك لماذا.")
    h2("محرك Autograd المرئي", "Autograd visualizer")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        w0 = st.slider("w", -3.0, 3.0, 3.0, 0.5, key="ag_w")
    with c2:
        b0 = st.slider("b", -2.0, 2.0, 0.5, 0.5, key="ag_b")
    with c3:
        x0 = st.slider("x", -3.0, 3.0, 2.0, 0.5, key="ag_x")
    with c4:
        y0 = st.slider("y", -3.0, 3.0, 1.0, 0.5, key="ag_y")
    T = torch()
    w = T.tensor(float(w0), requires_grad=True); b = T.tensor(float(b0), requires_grad=True); x = T.tensor(float(x0)); y = T.tensor(float(y0))
    y_hat = w * x + b; loss = (y_hat - y) ** 2; loss.backward()
    vals = dict(w=w0, b=b0, x=x0, y=y0, wx=w0 * x0, yhat=y_hat.item(), diff=(y_hat - y).item(), loss=loss.item(), g_diff=2 * (y_hat - y).item(), gw=w.grad.item(), gb=b.grad.item())
    caps = [
        ("الأوراق", f"**الأوراق**: `w` و`b` بـ `requires_grad=True` (ستُشتق)؛ `x` و`y` بيانات بلا تتبع. لا رسم بعد."),
        ("w·x", f"**عملية 1**: `w * x` = {vals['wx']:.2f}. الناتج يحمل `grad_fn=MulBackward` ويتذكر مدخليه — أول عقدة في الرسم."),
        ("ŷ", f"**عملية 2**: `+ b` → `ŷ` = {vals['yhat']:.2f} بـ `AddBackward`. الرسم يكبر ديناميكيًا مع كل سطر تكتبه."),
        ("ŷ−y", f"**عملية 3**: `ŷ − y` = {vals['diff']:.2f} (`SubBackward`)."),
        ("L", f"**عملية 4**: `(ŷ−y)²` → `L` = {vals['loss']:.2f} (`PowBackward`). انتهى الأمامي: الرسم كامل و`.grad` لا تزال `None`."),
        ("backward: pow", f"**`loss.backward()`** يبدأ من L بمشتقة 1. عقدة pow: ∂L/∂(ŷ−y) = 2(ŷ−y) = **{vals['g_diff']:.2f}**."),
        ("backward: sub", f"عقدة sub: ∂(ŷ−y)/∂ŷ = 1 ⇒ ∂L/∂ŷ = {vals['g_diff']:.2f}. (ولا تدرج نحو y: ليست ورقة مراقَبة.)"),
        ("backward: add", f"عقدة add: ∂ŷ/∂b = 1 ⇒ **b.grad = {vals['gb']:.2f}**؛ و∂ŷ/∂(w·x) = 1 يمرر {vals['g_diff']:.2f} إلى mul."),
        ("backward: mul", f"عقدة mul: ∂(w·x)/∂w = x = {x0:.2f} ⇒ **w.grad = {vals['g_diff']:.2f} × {x0:.2f} = {vals['gw']:.2f}**. (لا تدرج نحو x.)"),
        ("النتيجة", f"**التخزين**: `w.grad = {vals['gw']:.2f}`, `b.grad = {vals['gb']:.2f}` — في **الأوراق فقط**؛ الوسائط (ŷ) لا تحتفظ بتدرج. الرسم يُحرَّر بعد backward. لو نادينا backward ثانية **بلا تصفير** لأصبح w.grad = {2 * vals['gw']:.2f}."),
    ]
    frames = [Frame(_graph_svg(i, vals), caption(c), action=a, highlight=None) for i, (a, c) in enumerate(caps)]
    animation_player("ag_anim", frames, title_ar="الرسم الحسابي: أمامي يبني، خلفي يمشي", interval_ms=1500)
    equation(r"\frac{\partial L}{\partial w} = \underbrace{2(\hat y - y)}_{\text{pow}}\cdot\underbrace{1}_{\text{sub}}\cdot\underbrace{1}_{\text{add}}\cdot\underbrace{x}_{\text{mul}}, \qquad \frac{\partial L}{\partial b} = 2(\hat y - y)\cdot 1\cdot 1",
             [("grad_fn", "كل عقدة تعرف مشتقتها المحلية فقط."), (r"\text{backward}", "يضرب المشتقات المحلية من L إلى الورقة (قاعدة السلسلة، وحدة 5).")],
             meaning_ar="التدرج عند الورقة = حاصل ضرب المشتقات المحلية على المسار من الخسارة إليها.", example_ar=f"عند القيم الحالية: w.grad = {vals['g_diff']:.2f}×{x0:.2f} = {vals['gw']:.2f}.", dl_link_ar="لشبكة بملايين المعلمات نفس المشي، بمصفوفات (وحدة 14).", title_ar="ما يحسبه backward")
    code_lab(CodeLab(
        key="pt_autograd", title_ar="grad_fn، backward، التراكم، zero_، no_grad، detach", code=CODE, level="B",
        before=Before(goal_ar="مشاهدة الرسم يُبنى (grad_fn)، حساب التدرجات ومقارنتها باليد، إثبات التراكم عند backward متكرر، تصفير يدوي، خطوة تحديث تحت no_grad، وفصل موتر عن الرسم.", stage_ar="PyTorch: Autograd.",
                      inputs_ar="أعداد صغيرة: x=2, y=1, w=3, b=0.5.", expected_ar="w.grad = 22 وb.grad = 11 كما باليد؛ بعد backward ثانية 44 (تراكم)؛ بعد zero_ ثم backward 22؛ w يتحدث؛ pred.requires_grad False تحت no_grad.",
                      objects_ar="`requires_grad`, `grad_fn`, `.grad`, `backward()`, `.zero_()`, `torch.no_grad()`, `.detach()`."),
        explain=[("3-7", "الأوراق المراقَبة وبناء الرسم بأربع عمليات. `x` بلا requires_grad فلا `grad_fn` له."), ("8-10", "قبل backward: `.grad` هو `None` — لم يُحسب شيء بعد."),
                 ("11-13", "backward يملأ `.grad` للأوراق فقط؛ الوسائط (`y_hat.grad`) None. الأرقام تطابق اليد."), ("16-21", "**التراكم**: backward ثانية → 44 = 22 + 22. `.grad.zero_()` هو ما يفعله `optimizer.zero_grad()`."),
                 ("24-26", "التحديث اليدوي **داخل `no_grad`**: وإلا لسجّل Autograd التحديث كعملية في الرسم (وخطأ «leaf Variable in-place»)."), ("28-30", "`no_grad` للاستدلال: لا رسم، لا ذاكرة إضافية. `detach()` يعطي نسخة مفصولة عن الرسم (لـ `.numpy()` مثلًا).")],
        run=run_printed(CODE),
        after_ar="- `.grad` **يجمع** ولا يستبدل — هذه حقيقة تصميمية (مفيدة لتجميع التدرجات على عدة دفعات)، وسببُ `zero_grad`.\n- الأوراق فقط تحتفظ بتدرج؛ لطباعة تدرج وسيط استخدم `.retain_grad()`.\n- `no_grad` و`detach` هما طريقتا الخروج من الرسم: الأولى سياق، الثانية موتر.",
    ))
    h2("أين تُخزَّن التدرجات ولماذا؟", "Where gradients live & why")
    compare_table(["السؤال", "الجواب"],
                  [("أين؟", "في `.grad` لكل **ورقة** بـ `requires_grad=True` (المعلمات). موتر بنفس شكل المعلمة."), ("لماذا `.grad` وليس قيمة مرجعة؟", "لأن المحسّن يقرأها من المعلمات نفسها: `optimizer.step()` يمر على `model.parameters()` ويستخدم `p.grad`."),
                   ("لماذا تتراكم؟", "تصميم: يسمح بتجميع تدرجات عدة دفعات صغيرة قبل خطوة واحدة (دفعة فعلية أكبر بذاكرة أقل)، وبخسائر متعددة تُجمع تدرجاتها."), ("لماذا يجب المسح؟", "لأنك في الحلقة العادية تريد تدرج **هذه الدفعة فقط**؛ بلا مسح تحدّث بمجموع كل الدفعات السابقة — اتجاه خاطئ ومعيار متزايد (الصفحة السادسة)."),
                   ("متى يُحرَّر الرسم؟", "بعد `backward()` مباشرة (إلا `retain_graph=True`). لذلك تعيد التمرير الأمامي كل دفعة.")],
                  ["rtl", "rtl"])
    intuition("TensorFlow: «سجّل داخل الشريط، ثم اطلب التدرج كقيمة». PyTorch: «كل شيء مسجَّل، والتدرج يُودَع في المعلمة نفسها ويتراكم». الفرق الثاني هو ما يجعل حلقة PyTorch تبدأ بـ `zero_grad()`.")
    debugging_note("`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`: ناديت `backward()` على موتر لا يرتبط بأي ورقة مراقَبة — نموذج بلا `nn.Parameter`، أو بيانات مررت بـ `detach()`/`no_grad` قبل الخسارة، أو خسارة حُسبت بـ NumPy.")
    common_mistake("`w = w - lr * w.grad` (بلا `no_grad`) ينشئ موترًا **جديدًا** غير ورقة (له grad_fn)؛ في الدورة التالية `w.grad` يصبح None ويفشل التدريب بصمت. التحديث في مكانه تحت `no_grad` — أو دع `optimizer.step()` يفعله.")
    quiz("pt.autograd", [
        Q("`grad_fn` موجود على…", ["الأوراق", "نواتج العمليات على موترات مراقَبة", "كل الموترات"], 1, "يبني الرسم."),
        Q("بعد `backward()` مرتين بلا تصفير، `.grad` =", ["آخر تدرج", "مجموع التدرجين", "None"], 1, "تراكم."),
        Q("التدرجات تُخزَّن في…", ["قيمة مرجعة من backward", "`.grad` للأوراق المراقَبة", "الخسارة"], 1, "المحسّن يقرؤها."),
        Q("تحديث المعلمات يدويًا يجب أن يكون…", ["داخل الرسم", "تحت `torch.no_grad()` وفي مكانه", "بعد detach للخسارة"], 1, "لا تسجّل التحديث."),
    ])
    takeaway("الأمامي يبني الرسم (grad_fn)؛ backward يمشيه عكسيًا بقاعدة السلسلة ويجمع في .grad للأوراق. لذلك: zero_grad قبل backward، no_grad للتحديث والاستدلال، detach للخروج من الرسم.")
    lesson_footer(LESSON, ["المحرك المرئي: 10 خطوات.", "grad_fn و.grad والتراكم.", "no_grad وdetach."])
