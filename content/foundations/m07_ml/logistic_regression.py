import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation, worked_steps
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import pipeline

LESSON = Lesson(
    id="foundations.ml.logistic_regression",
    title_ar="جسر 2: الانحدار اللوجستي = خلية عصبية واحدة",
    title_en="Bridge 2: Logistic Regression = One Neuron",
    module="foundations.ml",
    order=6,
    prerequisites=["foundations.ml.linear_regression", "foundations.prob.likelihood", "foundations.calculus.chain_rule"],
    objectives_ar=[
        "بناء الانحدار اللوجستي: مجموع موزون ← Sigmoid ← احتمال ← عتبة ← فئة.",
        "اشتقاق تدرج BCE ورؤية أنه بنفس شكل تدرج MSE: $X^{\\mathsf T}(\\hat{p} - y)$.",
        "إدراك أن هذا النموذج هو خلية عصبية كاملة، والشبكة تراكب منها.",
    ],
    terms=["softmax", "cross_entropy", "weight", "bias", "probability"],
    labs=["labs.chain_rule_lab"],
    difficulty="intermediate",
    summary_ar="z = w·x + b، p = σ(z)، خسارة BCE، تدرج Xᵀ(p − y). هذه خلية عصبية؛ الشبكة تراكب منها.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(1)
n = 400
X = rng.normal(size=(n, 2))
true_w, true_b = np.array([2.0, -1.5]), 0.3
p_true = 1 / (1 + np.exp(-(X @ true_w + true_b)))
y = (rng.uniform(size=n) < p_true).astype(float)         # تسميات من احتمالات حقيقية

sigmoid = lambda z: 1 / (1 + np.exp(-z))
w, b = np.zeros(2), 0.0
eta = {eta}
for ep in range(1, 301):
    z = X @ w + b                                       # مجموع موزون   (n,)
    p = sigmoid(z)                                      # احتمال        (n,)
    loss = -np.mean(y*np.log(p + 1e-9) + (1-y)*np.log(1-p + 1e-9))   # BCE
    grad_w = X.T @ (p - y) / n                          # نفس شكل تدرج MSE!
    grad_b = (p - y).mean()
    w -= eta * grad_w; b -= eta * grad_b
    if ep in (1, 10, 50, 100, 300):
        acc = ((p >= 0.5) == y).mean()
        print(f"epoch {{ep:3d}}: BCE={{loss:.4f}} acc={{acc:.3f}} w={{w.round(2)}} b={{b:.2f}}")
print("true w =", true_w, " b =", true_b)
print("baseline acc (majority) =", max(y.mean(), 1 - y.mean()).round(3))'''


def _controls() -> dict:
    return {"eta": st.select_slider("η", options=[0.05, 0.1, 0.5, 1.0, 2.0], value=0.5, key="ctrl_logreg_eta")}


def render() -> None:
    lesson_header(LESSON)
    h2("من الخطي إلى الاحتمال", "From linear to probability")
    why("نريد التنبؤ بـ «هل يتعثر؟». الانحدار الخطي يعطي أي عدد (−3، 12) لا احتمالًا. نحتاج تحويلًا يضغط أي عدد إلى $(0, 1)$: هذا هو **Sigmoid**.")
    pipeline(["x (features)", "z = w·x + b", "p = σ(z)", "threshold 0.5", "class 0/1"], active=2)
    equation(r"z = \mathbf{w}\cdot\mathbf{x} + b, \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \hat{y} = \mathbb{1}[p \ge 0.5]",
             [("z", "المجموع الموزون (`logit`): أي عدد حقيقي."), (r"\sigma", "Sigmoid: تحوّل $z$ إلى احتمال؛ $\\sigma(0) = 0.5$، كبير موجب → 1، كبير سالب → 0."), ("p", "احتمال الفئة 1 كما يقدّره النموذج."), (r"\mathbb{1}[\cdot]", "العتبة: القرار النهائي (يمكن تغيير 0.5 حسب كلفة الأخطاء).")],
             meaning_ar="خط فاصل في فضاء الخصائص ($z = 0$)؛ بُعد النقطة عنه يحدد الثقة.",
             example_ar="$z = 2$: $p = 0.88$. $z = -2$: $p = 0.12$. $z = 0$: $p = 0.5$ على الحد الفاصل.",
             dl_link_ar="هذا **بالضبط** خلية عصبية: ضرب نقطي + انحياز + تنشيط. الشبكة العصبية = خلايا كثيرة في طبقات، والانحدار اللوجستي هو الشبكة الأصغر.", title_ar="الانحدار اللوجستي")
    intuition("الانحدار الخطي يرسم خطًا يمر **بين** النقاط؛ اللوجستي يرسم خطًا **يفصل** النقاط، ويحوّل المسافة عن الخط إلى احتمال.")
    h2("الخسارة والتدرج", "Loss & gradient")
    equation(r"L = -\frac{1}{n}\sum_i\big[y_i\log p_i + (1-y_i)\log(1-p_i)\big], \qquad \frac{\partial L}{\partial \mathbf{w}} = \frac{1}{n}X^{\mathsf T}(\mathbf{p} - \mathbf{y})",
             [("L", "الإنتروبيا المتقاطعة الثنائية = سالب لوغاريتم احتمال برنولي الأرجح (درس الاحتمال الأرجح)."), (r"\mathbf{p} - \mathbf{y}", "الخطأ الاحتمالي لكل ملاحظة: بين −1 و1."), (r"X^{\mathsf T}(\cdot)", "نفس بنية تدرج MSE.")],
             meaning_ar="مفاجأة: رغم اختلاف الخسارة والتنشيط، التدرج له نفس الشكل $X^{\\mathsf T}(\\text{تنبؤ} - \\text{حقيقة})$ لأن مشتقة Sigmoid تُلغي مشتقة اللوغاريتم بقاعدة السلسلة.",
             example_ar="ملاحظة بـ $y = 1$ و$p = 0.2$: الخطأ $-0.8$ يدفع الأوزان في اتجاه $+0.8\\,\\mathbf{x}$.",
             dl_link_ar="هذا الإلغاء الأنيق هو سبب أن Sigmoid+BCE وSoftmax+CCE أزواج «طبيعية»: تدرج طبقة الإخراج يصبح ببساطة (تنبؤ − حقيقة).", title_ar="BCE وتدرجها")
    worked_steps([("قاعدة السلسلة", r"\frac{\partial L_i}{\partial z_i} = \frac{\partial L_i}{\partial p_i}\cdot\frac{\partial p_i}{\partial z_i}"),
                  ("المشتقتان", r"\frac{\partial L_i}{\partial p_i} = -\frac{y_i}{p_i} + \frac{1-y_i}{1-p_i}, \qquad \frac{\partial p_i}{\partial z_i} = p_i(1-p_i)"),
                  ("الضرب والتبسيط", r"\frac{\partial L_i}{\partial z_i} = p_i - y_i"), ("ثم إلى الأوزان", r"\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{n}\sum_i (p_i - y_i)\mathbf{x}_i = \frac{1}{n}X^{\mathsf T}(\mathbf{p}-\mathbf{y})")],
                 title_ar="لماذا يبسط التدرج إلى p − y")
    code_lab(CodeLab(
        key="ml_logreg", title_ar="انحدار لوجستي من الصفر", code=CODE, template=True, defaults={"eta": 0.5},
        before=Before(goal_ar="تدريب خلية عصبية واحدة (انحدار لوجستي) بـ BCE والتدرج، ومقارنة المعلمات بالحقيقية والدقة بخط الأساس.", stage_ar="نموذج ← تدريب ← تقييم.",
                      inputs_ar="`X (400, 2)` وتسميات ثنائية مولّدة من احتمالات حقيقية.", expected_ar="BCE يهبط من 0.693 (= ln 2)، الدقة تتجاوز خط الأساس، و`w ≈ (2, −1.5)`, `b ≈ 0.3`."),
        explain=[("4-7", "نولّد التسميات **من احتمالات**: بعض النقاط قرب الحد ستُصنَّف خطأ حتى بالنموذج المثالي — حد أعلى للدقة."),
                 ("9-10", "Sigmoid وتهيئة بأصفار (z = 0 ⇒ p = 0.5 ⇒ خسارة ln 2 = 0.693 في البداية)."),
                 ("12-15", "أمامي: خطي ثم Sigmoid. الخسارة BCE مع `1e-9` لتفادي log(0)."),
                 ("16-18", "التدرج `X.T @ (p − y) / n` — قارنه بسطر الانحدار الخطي: متطابق البنية."),
                 ("21-23", "الدقة بعتبة 0.5، والمعلمات تقترب من الحقيقية.")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- الخسارة الابتدائية 0.693 = ln 2: القيمة المرجعية لمصنف ثنائي جاهل. إن بدأت أعلى بكثير فهناك خطأ في التهيئة أو التحجيم.\n- الدقة النهائية ليست 100% ولا يجب أن تكون: التسميات نفسها احتمالية.",
    ))
    h3("حد القرار", "Decision boundary")
    rng = np.random.default_rng(1); Xp = rng.normal(size=(300, 2)); pp = 1 / (1 + np.exp(-(Xp @ np.array([2.0, -1.5]) + 0.3))); yp = (rng.uniform(size=300) < pp)
    xs = np.linspace(-3, 3, 50)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Xp[yp, 0], y=Xp[yp, 1], mode="markers", name="y = 1", marker=dict(color="#C8473A", size=7)))
    fig.add_trace(go.Scatter(x=Xp[~yp, 0], y=Xp[~yp, 1], mode="markers", name="y = 0", marker=dict(color="#2F6FB5", size=7)))
    fig.add_trace(go.Scatter(x=xs, y=(-0.3 - 2.0 * xs) / (-1.5), mode="lines", name="z = 0 (p = 0.5)", line=dict(color="#1F7A78", width=3)))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), xaxis=dict(range=[-3, 3], title="x₁"), yaxis=dict(range=[-3, 3], title="x₂"), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="logreg_boundary")
    st.caption("خلية واحدة = حد قرار **خطي** واحد. النقاط المختلطة قرب الخط لا يمكن فصلها بخط؛ ولفصل أشكال منحنية نحتاج طبقات مخفية لاخطية — موضوع الوحدات التالية.")
    common_mistake("قراءة `p = 0.5` كـ «النموذج غير متأكد بنسبة 50%» ثم التوقف. العتبة 0.5 افتراضية؛ إذا كانت كلفة تفويت متعثر أعلى من كلفة إزعاج عميل جيد فاخفضها (0.3 مثلًا). العتبة قرار اقتصادي لا رياضي.")
    quiz("ml.logreg", [
        Q("z = 3: p ≈", ["0.95", "0.5", "0.05"], 0, "σ(3) ≈ 0.953.", kind="equation"),
        Q("تدرج BCE بالنسبة لـ z…", ["p(1−p)", "p − y", "y/p"], 1, "الإلغاء الأنيق."),
        Q("خسارة BCE الابتدائية بأوزان صفرية…", ["0", "0.693", "1"], 1, "ln 2."),
        Q("الانحدار اللوجستي هو…", ["شبكة بطبقتين", "خلية عصبية واحدة بتنشيط Sigmoid", "انحدار خطي بعتبة"], 1, "الجسر."),
    ])
    takeaway("خلية = w·x + b → σ → p. BCE من الاحتمال الأرجح؛ تدرجها Xᵀ(p − y). حد قرار خطي واحد؛ الطبقات المخفية تكسر الخطية.")
    lesson_footer(LESSON, ["Sigmoid يضغط z إلى احتمال.", "∂L/∂z = p − y.", "العتبة قرار اقتصادي.", "خلية واحدة = حد خطي."])
