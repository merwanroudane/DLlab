import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto

LESSON = Lesson(
    id="foundations.optim.gd_variants",
    title_ar="الانحدار التدريجي: الكامل، العشوائي، والدفعات الصغيرة",
    title_en="Gradient Descent: Batch, Stochastic & Mini-batch",
    module="foundations.optim",
    order=2,
    prerequisites=["foundations.optim.loss_surface", "foundations.ml.linear_regression"],
    objectives_ar=["قاعدة التحديث الواحدة والفرق الوحيد بين الأنواع الثلاثة: على كم ملاحظة يُحسب التدرج.", "فهم مقايضة الضوضاء/الكلفة، ولماذا الدفعات الصغيرة هي الافتراضي.", "قياس الفرق بالكود."],
    terms=["batch", "batch_size", "epoch", "iteration"],
    labs=["labs.gradient_descent_lab", "labs.epoch_batch_simulator"],
    difficulty="intermediate",
    summary_ar="نفس القاعدة θ ← θ − η∇L؛ الفرق حجم العينة التي يُقدَّر منها ∇L: كل البيانات، ملاحظة واحدة، أو دفعة. الدفعات = ضوضاء مفيدة بكلفة معقولة.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(0)
n = 512
X = rng.normal(size=(n, 3)); w_true = np.array([2.0, -1.0, 0.5])
y = X @ w_true + rng.normal(0, 0.3, n)

def run(batch_size, eta, epochs):
    w = np.zeros(3); steps = 0; hist = []
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch_size):
            b = idx[s:s+batch_size]
            g = 2 * X[b].T @ (X[b] @ w - y[b]) / len(b)      # تدرج على العينة
            w -= eta * g; steps += 1
        hist.append(np.mean((X @ w - y)**2))                  # خسارة كل البيانات
    return w, steps, hist

for name, bs in [("batch GD  ", n), ("mini-batch", 32), ("SGD (1)   ", 1)]:
    w, steps, hist = run(bs, {eta}, {epochs})
    print(f"{{name}} bs={{bs:>3}} updates={{steps:>5}}  final MSE={{hist[-1]:.4f}}  MSE after epoch 1={{hist[0]:.4f}}  w={{w.round(2)}}")'''


def _controls() -> dict:
    c1, c2 = st.columns(2)
    with c1:
        eta = st.select_slider("η", options=[0.001, 0.01, 0.05, 0.1], value=0.05, key="ctrl_gdv_eta")
    with c2:
        epochs = st.slider("epochs", 1, 20, 5, key="ctrl_gdv_epochs")
    return {"eta": eta, "epochs": epochs}


def render() -> None:
    lesson_header(LESSON)
    h2("قاعدة واحدة، ثلاث عينات", "One rule, three sample sizes")
    equation(r"\theta \leftarrow \theta - \eta\,\hat{\nabla} L(\theta), \qquad \hat\nabla L = \frac{1}{|B|}\sum_{i \in B}\nabla \ell_i(\theta)",
             [(r"\hat\nabla L", "تقدير التدرج من العينة $B$."), ("|B| = n", "كل البيانات: **Batch GD** — تدرج دقيق، خطوة واحدة لكل حقبة، مكلف للبيانات الكبيرة."), ("|B| = 1", "ملاحظة واحدة: **SGD** خالص — تدرج ضجيج جدًا، تحديثات كثيرة رخيصة."), ("1 < |B| < n", "**Mini-batch** — الافتراضي: توازن، ويستغل توازي GPU.")],
             meaning_ar="الأنواع الثلاثة نفس الخوارزمية؛ يختلفون في دقة تقدير التدرج وعدد التحديثات لكل حقبة.",
             example_ar="n = 512، دفعة 32 ⇒ 16 تحديثًا في الحقبة. دفعة 512 ⇒ تحديث واحد. دفعة 1 ⇒ 512.",
             dl_link_ar="`batch_size` في `fit` هو |B|. عمليًا «SGD» في الأطر يعني mini-batch SGD.", title_ar="الانحدار التدريجي بعينات")
    intuition("Batch GD يسأل كل الناخبين قبل كل قرار: دقيق وبطيء. SGD يسأل شخصًا واحدًا: سريع ومتقلب. Mini-batch يسأل مجموعة تركيز: قرار معقول بسرعة معقولة — والتقلب الصغير يساعد على الهروب من السروج.")
    compare_table(["", "Batch GD", "Mini-batch", "SGD"],
                  [("|B|", "n", "32–512", "1"), ("ضوضاء التدرج", "لا", "متوسطة", "عالية"), ("تحديثات/حقبة", "1", "n/|B|", "n"), ("كلفة التحديث", "عالية", "متوسطة", "منخفضة"),
                   ("منحنى الخسارة", "ناعم", "متذبذب قليلًا", "متذبذب جدًا"), ("الذاكرة", "كل البيانات", "دفعة", "ملاحظة"), ("GPU", "غير عملي", "مثالي", "غير مستغل"), ("الهروب من السروج", "ضعيف", "جيد", "جيد جدًا")],
                  ["rtl", "rtl", "rtl", "rtl"])
    code_lab(CodeLab(
        key="optim_gdv", title_ar="الأنواع الثلاثة على نفس المسألة", code=CODE, template=True, defaults={"eta": 0.05, "epochs": 5},
        before=Before(goal_ar="تدريب انحدار خطي بثلاثة أحجام دفعات ومقارنة عدد التحديثات والخسارة بعد الحقبة الأولى وفي النهاية.", stage_ar="التحسين.",
                      inputs_ar="512 ملاحظة بثلاث خصائص.", expected_ar="بعد الحقبة الأولى Batch GD ما زال بعيدًا (تحديث واحد) بينما mini-batch وSGD تقدما كثيرًا؛ في النهاية الثلاثة قريبون من w_true."),
        explain=[("7-15", "دالة واحدة؛ الفرق فقط في `batch_size`. الخسارة تُقاس على كل البيانات في نهاية كل حقبة (للمقارنة العادلة)."), ("17-19", "ثلاث تجارب بنفس η والحقب.")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- بنفس عدد الحقب، عدد **التحديثات** يختلف بـ 512 ضعفًا. لهذا تتقدم الدفعات الصغيرة أسرع لكل حقبة.\n- مع η = 0.1 قد يتذبذب SGD أكثر: الضوضاء + خطوة كبيرة. الدفعات الأكبر تحتمل η أكبر.",
    ))
    common_mistake("مقارنة محسّنين بعدد الحقب فقط دون عدد التحديثات أو الزمن الفعلي. حقبة بدفعة 8 تعمل 64 ضعف تحديثات حقبة بدفعة 512.")
    st.button("افتح معمل الانحدار التدريجي", icon=":material/science:", on_click=goto, args=("labs.gradient_descent_lab",), key="gdv_lab")
    quiz("optim.gdv", [
        Q("n = 1000، batch_size = 50: تحديثات كل حقبة…", ["1", "20", "1000"], 1, "n/|B|."),
        Q("منحنى خسارة متذبذب جدًا بدفعة 1 يدل غالبًا على…", ["خطأ في الكود", "ضوضاء تقدير التدرج", "تسريب"], 1, "طبيعي لـ SGD."),
        Q("لماذا mini-batch هو الافتراضي؟", ["أدق تدرج", "توازن الضوضاء والكلفة ويستغل GPU", "أقل ذاكرة"], 1, "المقايضة."),
    ])
    takeaway("قاعدة واحدة؛ الفرق حجم العينة. الدفعات الصغيرة: تحديثات كثيرة، ضوضاء مفيدة، توازي. قارن بالتحديثات لا بالحقب.")
    lesson_footer(LESSON, ["|B| يحدد الضوضاء والكلفة.", "عدد التحديثات = n/|B| لكل حقبة.", "الضوضاء تساعد على الهروب."])
