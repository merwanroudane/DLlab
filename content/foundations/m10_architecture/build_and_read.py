import numpy as np
import streamlit as st

from components.callouts import common_mistake, practical_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go
from core.rtl import table

LESSON = Lesson(
    id="foundations.architecture.build_and_read",
    title_ar="بناء شبكة كثيفة في NumPy وقراءة أشكالها",
    title_en="Building a Dense Network in NumPy & Reading its Shapes",
    module="foundations.architecture",
    order=4,
    prerequisites=["foundations.architecture.parameter_count", "foundations.linalg.tensors"],
    objectives_ar=["بناء شبكة كثيفة كاملة كقائمة طبقات في NumPy مع تهيئة عشوائية.", "تتبع شكل الدفعة عبر كل طبقة وطباعة ملخص يشبه `model.summary()`.", "فهم لماذا التهيئة عشوائية ولماذا ليست كبيرة."],
    terms=["seed", "shape", "batch_dimension"],
    labs=["labs.network_builder"],
    difficulty="intermediate",
    summary_ar="شبكة = قائمة (W, b, f). التمرير عبرها يغيّر المحور الأخير فقط؛ الدفعة تبقى. التهيئة عشوائية صغيرة.",
)

CODE = '''import numpy as np
rng = np.random.default_rng({seed})

def relu(z): return np.maximum(0, z)
def sigmoid(z): return 1 / (1 + np.exp(-z))

def build(sizes, activations, scale=0.1):
    """sizes = [n_in, h1, ..., n_out]; يعيد قائمة طبقات (W, b, f)."""
    layers = []
    for n_in, n_out, f in zip(sizes[:-1], sizes[1:], activations):
        W = rng.normal(0, scale, (n_in, n_out))     # تهيئة عشوائية صغيرة
        b = np.zeros(n_out)
        layers.append((W, b, f))
    return layers

def forward(layers, X, verbose=True):
    a = X
    if verbose: print(f"input           shape={{a.shape}}")
    for i, (W, b, f) in enumerate(layers, 1):
        z = a @ W + b
        a = f(z)
        if verbose: print(f"dense_{{i}} ({{f.__name__:<7}}) W{{W.shape}} b{{b.shape}} -> out shape={{a.shape}}  params={{W.size + b.size}}")
    return a

net = build([{n_in}, 16, 8, 1], [relu, relu, sigmoid])
X = rng.normal(size=({batch}, {n_in}))
p = forward(net, X)
print("total params =", sum(W.size + b.size for W, b, _ in net))
print("first 3 predictions:", p[:3, 0].round(3), " (all near 0.5 before training)")

# لماذا التهيئة الصغيرة؟ جرّب مقياسًا كبيرًا:
big = build([{n_in}, 16, 8, 1], [relu, relu, sigmoid], scale=3.0)
print("large init predictions:", forward(big, X, verbose=False)[:3, 0].round(3), " (saturated: 0 or 1 -> zero gradient)")'''


def _controls() -> dict:
    c1, c2, c3 = st.columns(3)
    with c1:
        n_in = st.slider("n_in", 2, 20, 4, key="ctrl_bar_nin")
    with c2:
        batch = st.slider("batch", 1, 64, 5, key="ctrl_bar_batch")
    with c3:
        seed = st.number_input("seed", 0, 99, 0, key="ctrl_bar_seed")
    return {"n_in": n_in, "batch": batch, "seed": seed}


def render() -> None:
    lesson_header(LESSON)
    h2("الشبكة كقائمة طبقات", "A network as a list of layers")
    code_lab(CodeLab(
        key="arch_build", title_ar="بناء وتمرير وملخص", code=CODE, template=True, defaults={"n_in": 4, "batch": 5, "seed": 0},
        before=Before(goal_ar="بناء شبكة n_in → 16 → 8 → 1 من الصفر، تمرير دفعة، وطباعة الأشكال والمعلمات كملخص.", stage_ar="البنية ← التمرير الأمامي (تمهيد).",
                      inputs_ar="دفعة عشوائية `(batch, n_in)`.", expected_ar="أشكال متسلسلة `(batch, 16) → (batch, 8) → (batch, 1)`، مجموع المعلمات، تنبؤات ≈ 0.5، ثم تنبؤات مشبعة مع تهيئة كبيرة."),
        explain=[("7-14", "`build` ينشئ لكل طبقة W بشكل (in, out) وb أصفارًا وتنشيطًا. التهيئة `normal(0, 0.1)`."),
                 ("16-23", "`forward` يمرر الدفعة طبقة طبقة: `a @ W + b` ثم `f`. لاحظ أن المحور 0 (الدفعة) لا يتغير أبدًا."),
                 ("25-29", "بنية بثلاث طبقات وتمريرة. قبل التدريب المخرج ≈ 0.5 لأن الأوزان صغيرة فـ z ≈ 0."),
                 ("32-33", "تهيئة كبيرة (3.0) تعطي z ضخمة فتشبع Sigmoid عند 0 أو 1 — مشتقتها هناك ≈ 0 فلا يتعلم النموذج. لذلك التهيئة صغيرة (ومحسوبة: Glorot/He في الأطر).")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- الملخص المطبوع يحاكي `model.summary()`: اسم الطبقة، الأشكال، المعلمات.\n- التهيئة العشوائية ضرورية (وليست صفرًا) حتى تختلف الخلايا عن بعضها؛ لو بدأت كلها بصفر لتعلمت نفس الشيء (كسر التناظر).",
    ))
    table(["السؤال", "الجواب من الملخص"],
          [("كم مدخلًا تتوقع الشبكة؟", "n_in = W₁.shape[0]"), ("ما شكل المخرج لدفعة 32؟", "(32, آخر n_out)"), ("كم معلمة؟", "Σ (W.size + b.size)"), ("أين تكمن معظم المعلمات؟", "في الطبقة ذات أكبر in×out — غالبًا الأولى بعد Flatten في الصور")],
          ["rtl", "code"])
    practical_note("لماذا لا نبدأ بأوزان صفرية؟ كل خلايا الطبقة ستحسب نفس الشيء وتتلقى نفس التدرج فتبقى متطابقة إلى الأبد. العشوائية «تكسر التناظر». ولماذا صغيرة؟ لتجنب إشباع التنشيطات كما رأيت.")
    st.button("افتح معمل بناء الشبكة (على بيانات حقيقية)", icon=":material/science:", type="primary", on_click=go, args=("labs.network_builder",), key="build_lab_btn")
    common_mistake("تمرير مدخل بالشكل `(n_in,)` بدل `(1, n_in)`: `a @ W` ينجح لكن الانحياز والطبقات التالية تعطي أشكالًا غير متوقعة. حافظ على محور الدفعة دائمًا.")
    quiz("arch.build", [
        Q("دفعة `(32, 10)` عبر شبكة 10 → 64 → 3: شكل المخرج…", ["(32, 3)", "(3,)", "(64, 3)"], 0, "الدفعة تبقى.", kind="shape"),
        Q("لماذا التهيئة عشوائية لا صفرية؟", ["للسرعة", "لكسر التناظر بين الخلايا", "لأن NumPy يفرض ذلك"], 1, "خلايا مختلفة."),
        Q("تهيئة بمقياس كبير مع Sigmoid تسبب…", ["تعلمًا أسرع", "إشباعًا وتدرجًا صفريًا", "لا شيء"], 1, "z ضخمة."),
    ])
    takeaway("الشبكة قائمة (W, b, f). التمرير يغيّر المحور الأخير فقط. تهيئة عشوائية صغيرة: عشوائية لكسر التناظر، صغيرة لتجنب الإشباع.")
    lesson_footer(LESSON, ["build/forward في 20 سطرًا.", "الملخص يجيب عن أسئلة الشكل والمعلمات.", "التهيئة معلمة تصميم لها أثر كبير."])
