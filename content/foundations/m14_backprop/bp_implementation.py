import streamlit as st

from components.callouts import common_mistake, practical_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.backprop.backpropagation.implementation",
    title_ar="التنفيذ في بايثون: backward كامل + تحقق عددي",
    title_en="Python Implementation: a Full backward() + Gradient Check",
    module="foundations.backprop",
    parent="foundations.backprop.backpropagation",
    order=5,
    prerequisites=["foundations.backprop.backpropagation.two_layer"],
    objectives_ar=["كتابة `backward` عامة لأي شبكة كثيفة تُعيد تدرجات بأشكال المعلمات.", "التحقق العددي من كل تدرج بالفروق المنتهية — الاختبار الذي يكشف أي خطأ.", "تدريب الشبكة بضع خطوات ورؤية الخسارة تهبط."],
    terms=["backpropagation", "gradient"],
    labs=["labs.network_builder"],
    difficulty="advanced",
    summary_ar="forward يحفظ، backward يعود بالقواعد الثلاث، gradient check يؤكد، ثم حلقة تدريب صغيرة.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(0)
relu, drelu = (lambda z: np.maximum(0, z)), (lambda z: (z > 0).astype(float))
sigmoid = lambda z: 1/(1+np.exp(-z))

def init(sizes):
    return [(rng.normal(0, np.sqrt(2/i), (i, o)), np.zeros(o)) for i, o in zip(sizes[:-1], sizes[1:])]

def forward(X, params):
    a, cache = X, {"a": [X], "z": []}
    for l, (W, b) in enumerate(params):
        z = a @ W + b; cache["z"].append(z)
        a = sigmoid(z) if l == len(params)-1 else relu(z); cache["a"].append(a)
    return a[:, 0], cache

def loss(p, y): p = np.clip(p, 1e-7, 1-1e-7); return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

def backward(cache, y, params):
    n = len(y); grads = [None]*len(params)
    delta = (cache["a"][-1][:, 0] - y)[:, None] / n                # δ_L = (p − y)/n  (Sigmoid+BCE)
    for l in reversed(range(len(params))):
        W, b = params[l]
        gW = cache["a"][l].T @ delta                              # a_prevᵀ δ
        gb = delta.sum(0)                                         # Σ δ
        grads[l] = (gW, gb)
        assert gW.shape == W.shape and gb.shape == b.shape
        if l > 0:
            delta = (delta @ W.T) * drelu(cache["z"][l-1])        # (δ Wᵀ) ⊙ f'(z)
    return grads

# بيانات صغيرة وشبكة 2 → 8 → 4 → 1
X = rng.normal(size=(64, 2)); y = ((X[:, 0]**2 + X[:, 1]**2) > 1.0).astype(float)
params = init([2, 8, 4, 1])

# تحقق عددي: عنصر واحد من كل مصفوفة
p, cache = forward(X, params); grads = backward(cache, y, params)
for l, (W, b) in enumerate(params):
    i, j = 0, 0; eps = 1e-5
    W[i, j] += eps; lp = loss(forward(X, params)[0], y); W[i, j] -= 2*eps; lm = loss(forward(X, params)[0], y); W[i, j] += eps
    num = (lp - lm) / (2*eps); ana = grads[l][0][i, j]
    print(f"layer {l}: analytic={ana:+.6f} numeric={num:+.6f} rel.err={abs(ana-num)/max(abs(num),1e-12):.1e}")

# بضع خطوات تدريب
eta = {eta}
for step in range(1, 201):
    p, cache = forward(X, params); grads = backward(cache, y, params)
    for (W, b), (gW, gb) in zip(params, grads):
        W -= eta*gW; b -= eta*gb
    if step in (1, 10, 50, 100, 200):
        print(f"step {{step:3d}}: loss={{loss(p, y):.4f}} acc={{((p>=0.5)==y).mean():.3f}}")'''


def _controls() -> dict:
    return {"eta": st.select_slider("η", options=[0.05, 0.1, 0.3, 0.5, 1.0], value=0.3, key="ctrl_bpimpl_eta")}


def render() -> None:
    lesson_header(LESSON)
    h2("الكود الكامل", "The complete code")
    code_lab(CodeLab(
        key="bp_impl", title_ar="forward / backward / gradient check / train", code=CODE, template=True, defaults={"eta": 0.3},
        before=Before(goal_ar="تنفيذ الانتشار الخلفي لشبكة 2 → 8 → 4 → 1، التحقق منه عدديًا لكل طبقة، ثم تدريب 200 خطوة.", stage_ar="الانتشار الخلفي ← التنفيذ.",
                      inputs_ar="64 نقطة ثنائية البُعد وهدف «داخل/خارج دائرة».", expected_ar="خطأ نسبي ~1e-6 أو أقل لكل طبقة (التحقق ناجح)، ثم خسارة تهبط ودقة ترتفع.",
                      math_ar="$\\delta^{(L)} = (p - y)/n$، $g_W = a^{\\mathsf T}\\delta$، $\\delta \\leftarrow (\\delta W^{\\mathsf T}) \\odot f'(z)$."),
        explain=[("9-14", "`forward` يحفظ كل a وz — بدونها لا خلفي."), ("18-28", "`backward`: يبدأ من δ_L، ولكل طبقة من الأخيرة: تدرج W وb، `assert` على الأشكال، ثم يمرر δ إلى الخلف."),
                 ("35-40", "التحقق العددي: نغيّر وزنًا واحدًا بمقدار ±ε ونقيس تغير الخسارة. الخطأ النسبي يجب أن يكون < 1e-5. هذا الاختبار **إلزامي** لأي تنفيذ يدوي."),
                 ("43-49", "حلقة تدريب بأبسط محسّن؛ الأمامي والخلفي ثم التحديث.")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- خطأ نسبي بمقدار 1e-6 إلى 1e-9: التنفيذ صحيح. خطأ 1e-2 أو أكبر: خلل (غالبًا نسيان f′ أو خطأ في المنقولة).\n- `assert` على الأشكال يمنع أشهر خطأ قبل أن يظهر كنتائج غريبة.",
    ))
    table(["الخطأ المحتمل", "العرض في التحقق العددي", "الإصلاح"],
          [("نسيان ⊙ f'(z)", "خطأ كبير في كل الطبقات ما عدا الأخيرة", "أضف مشتقة التنشيط عند العودة"), ("استخدام W بدل Wᵀ", "خطأ شكل (يفشل assert)", "delta @ W.T"),
           ("نسيان القسمة على n", "تدرج أكبر بمعامل n", "δ_L = (p − y)/n"), ("استخدام a بدل z في f'", "خطأ متوسط", "f'(z) لا f'(a) (للـ Sigmoid تُكتب بدلالة a عمدًا)")],
          ["rtl", "rtl", "code"])
    practical_note("في الأطر لن تكتب هذا. لكن عندما تبني طبقة مخصصة أو خسارة مخصصة، أو تشك في تدرج، فالتحقق العددي هو نفس الأداة: `tf.test.compute_gradient` و`torch.autograd.gradcheck`.")
    common_mistake("تشغيل التحقق العددي بـ ε كبير (1e-2) أو صغير جدًا (1e-10): الأول تقريب سيئ، الثاني ضوضاء تمثيل عشري. 1e-5 مع float64 مناسب.")
    quiz("bp.impl", [
        Q("خطأ نسبي 3e-2 في التحقق العددي للطبقة الأولى فقط…", ["طبيعي", "خلل: غالبًا نسيان f' عند العودة", "ε كبير"], 1, "الطبقة الأخيرة لا تحتاج f' الخلفية."),
        Q("`assert gW.shape == W.shape` يكشف…", ["خطأ القيمة", "خطأ المنقولة/الأشكال", "خطأ معدل التعلم"], 1, "الأشكال."),
        Q("مقابل التحقق العددي في PyTorch…", ["torch.no_grad", "torch.autograd.gradcheck", "optimizer.step"], 1, "نفس الفكرة."),
    ])
    takeaway("backward عام في 12 سطرًا. التحقق العددي إلزامي: خطأ نسبي < 1e-5 أو هناك خلل. الأشكال أول assert.")
    lesson_footer(LESSON, ["forward يحفظ، backward يعود.", "gradient check أداة الثقة.", "جدول الأخطاء وكيف تظهر."])
