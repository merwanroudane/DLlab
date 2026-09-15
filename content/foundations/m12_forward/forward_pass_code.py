import numpy as np
import streamlit as st

from components.callouts import common_mistake, practical_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.forward.forward_pass_code",
    title_ar="التمرير الأمامي كودًا: MLP في NumPy مع تتبع الأشكال",
    title_en="Forward Pass in Code: a NumPy MLP with Shape Tracing",
    module="foundations.forward",
    order=4,
    prerequisites=["foundations.forward.numerical_walkthrough", "foundations.architecture.build_and_read"],
    objectives_ar=["كتابة دالة `forward` عامة تحفظ الوسائط.", "تشغيلها خطوة بخطوة مع طباعة الشكل والإحصاءات عند كل طبقة.", "ربط كل سطر بمعادلته وبمقابله في Keras وPyTorch."],
    terms=["shape", "batch_dimension", "softmax"],
    labs=["labs.network_builder"],
    difficulty="intermediate",
    summary_ar="forward = حلقة على الطبقات تحفظ z وa؛ نفس الشيء الذي يفعله model(X) في أي إطار.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(0)
relu = lambda z: np.maximum(0, z)
def softmax(z): e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

sizes = [{d}, 16, 8, 3]                                   # 3 فئات
params = [(rng.normal(0, np.sqrt(2/i), (i, o)), np.zeros(o)) for i, o in zip(sizes[:-1], sizes[1:])]
acts = [relu, relu, softmax]

def forward(X, params, acts, trace=False):
    cache = {"a0": X}
    a = X
    for l, ((W, b), f) in enumerate(zip(params, acts), start=1):
        z = a @ W + b
        a = f(z)
        cache[f"z{l}"], cache[f"a{l}"] = z, a
        if trace:
            zero = (a == 0).mean() if f is relu else 0.0
            print(f"layer {l}: z{z.shape} mean={z.mean():+.3f} std={z.std():.3f} | a{a.shape} mean={a.mean():.3f} zero-frac={zero:.2f}")
    return a, cache

X = rng.normal(size=({n}, {d})).astype(np.float32)
probs, cache = forward(X, params, acts, trace=True)
print("probs shape:", probs.shape, " row sums:", probs.sum(1)[:3].round(3))
print("predicted classes:", probs.argmax(1)[:8])
print("cached tensors:", {k: v.shape for k, v in cache.items()})'''


def _controls() -> dict:
    c1, c2 = st.columns(2)
    with c1:
        n = st.slider("n (دفعة)", 1, 128, 32, key="ctrl_fwdc_n")
    with c2:
        d = st.slider("d (خصائص)", 2, 32, 6, key="ctrl_fwdc_d")
    return {"n": n, "d": d}


def render() -> None:
    lesson_header(LESSON)
    h2("الدالة العامة", "The general function")
    code_lab(CodeLab(
        key="fwd_code", title_ar="forward مع ذاكرة وتتبع", code=CODE, template=True, defaults={"n": 32, "d": 6},
        before=Before(goal_ar="تنفيذ تمرير أمامي لشبكة d → 16 → 8 → 3 مع حفظ z وa لكل طبقة وطباعة تشخيص عند كل طبقة.", stage_ar="التمرير الأمامي ← الكود.",
                      inputs_ar="دفعة `(n, d)` عشوائية.", expected_ar="ثلاثة أسطر تتبع بأشكال `(n,16)`, `(n,8)`, `(n,3)`، مجاميع صفوف = 1، فئات، وقاموس الذاكرة.",
                      math_ar="$\\mathbf{z}^{(\\ell)} = \\mathbf{a}^{(\\ell-1)}W^{(\\ell)} + \\mathbf{b}^{(\\ell)}$، $\\mathbf{a}^{(\\ell)} = f^{(\\ell)}(\\mathbf{z}^{(\\ell)})$."),
        explain=[("6-8", "بنية وتهيئة He (`√(2/fan_in)`) وقائمة تنشيطات: ReLU للمخفية وSoftmax للإخراج."),
                 ("10-20", "الحلقة: ضرب، انحياز، تنشيط، حفظ. `cache` هو ما ستحتاجه دالة `backward` في الوحدة 14. التتبع يطبع إحصاءات z وa ونسبة الأصفار (كشف مبكر لـ Dead ReLU)."),
                 ("22-26", "التشغيل على دفعة: Softmax يعطي مجموع 1 لكل صف؛ argmax الفئات. لاحظ أن `cache` يحوي `(n, 16)+(n, 8)+(n, 3)` قيمًا لكل من z وa — ذاكرة تتناسب مع حجم الدفعة.")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- `zero-frac ≈ 0.5` في الطبقات المخفية طبيعي مع ReLU وتهيئة He (نصف الوحدات نشطة).\n- الاحتمالات قبل التدريب ≈ 1/3 لكل فئة: شبكة جاهلة، وهذا صحيح.",
    ))
    h2("نفس الشيء في الأطر", "The same thing in frameworks")
    table(["الخطوة", "NumPy (هنا)", "Keras", "PyTorch"],
          [("طبقة كثيفة", "a @ W + b", "Dense(16)", "nn.Linear(d, 16)"), ("تنشيط", "relu(z)", "activation='relu'", "F.relu / nn.ReLU()"), ("الإخراج", "softmax(z)", "Dense(3, activation='softmax')", "nn.Linear(8, 3) + F.softmax (أو داخل الخسارة)"),
           ("التمرير", "forward(X, ...)", "model(X) / model.predict(X)", "model(X) (forward)"), ("حفظ الوسائط", "cache", "تلقائي داخل GradientTape", "تلقائي عبر autograd")],
          ["rtl", "code", "code", "code"])
    practical_note("في الأطر لا تكتب `cache` بنفسك: الاشتقاق التلقائي يسجّل العمليات. لكن الفهم هنا يجعل `GradientTape` و`loss.backward()` بديهيين لاحقًا.")
    common_mistake("تمرير `X` من نوع `float64` إلى إطار يتوقع `float32`، أو بلا محور دفعة. الدالة هنا لا تشتكي؛ الأطر تشتكي — ولذلك نتحقق من الشكل والنوع أولًا.")
    quiz("fwd.code", [
        Q("`cache['a2'].shape` لدفعة 32 عبر 6 → 16 → 8 → 3…", ["(32, 8)", "(8,)", "(32, 3)"], 0, "مخرج الطبقة الثانية.", kind="shape"),
        Q("لماذا نحفظ z وa؟", ["للطباعة", "للانتشار الخلفي", "للسرعة"], 1, "قاعدة السلسلة تحتاجها."),
        Q("مقابل `a @ W + b` في PyTorch…", ["nn.Linear", "nn.ReLU", "torch.softmax"], 0, "الطبقة الخطية."),
    ])
    takeaway("forward = حلقة تحفظ الوسائط. الأطر تفعل الشيء نفسه وتسجل العمليات تلقائيًا. التتبع عند كل طبقة أداة تشخيص مجانية.")
    lesson_footer(LESSON, ["دالة عامة لأي بنية كثيفة.", "cache للخلفي.", "جدول المقابلات Keras/PyTorch."])
