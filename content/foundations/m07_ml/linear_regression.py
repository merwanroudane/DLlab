import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from labs.datasets import study_hours

LESSON = Lesson(
    id="foundations.ml.linear_regression",
    title_ar="جسر 1: الانحدار الخطي بالانحدار التدريجي",
    title_en="Bridge 1: Linear Regression by Gradient Descent",
    module="foundations.ml",
    order=5,
    prerequisites=["foundations.ml.baseline_generalization", "foundations.calculus.partial_gradient", "foundations.linalg.matrix_multiplication"],
    objectives_ar=[
        "كتابة الانحدار الخطي متعدد المدخلات بصيغة المصفوفات $\\hat{y} = Xw + b$.",
        "اشتقاق تدرج MSE بالنسبة لـ w وb وتنفيذ الانحدار التدريجي من الصفر.",
        "مقارنة الحل بالانحدار التدريجي مع الحل التحليلي (المربعات الصغرى).",
    ],
    terms=["weight", "bias", "gradient", "learning_rate", "loss"],
    labs=["labs.gradient_lab", "labs.epoch_batch_simulator"],
    difficulty="intermediate",
    summary_ar="الانحدار الخطي = طبقة كثيفة واحدة بلا تنشيط، تُدرَّب بتقليل MSE بالتدرج؛ نفس الحلقة التي تدرّب أي شبكة.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(0)
n = 200
X = rng.normal(size=(n, 2))                       # خاصيتان (موحدتان)
true_w, true_b = np.array([3.0, -2.0]), 5.0
y = X @ true_w + true_b + rng.normal(0, 0.5, n)   # هدف + ضوضاء

w, b = np.zeros(2), 0.0                           # تهيئة
eta, epochs = {eta}, {epochs}
for ep in range(1, epochs + 1):
    y_hat = X @ w + b                             # تمرير أمامي (n,)
    err = y_hat - y
    loss = np.mean(err ** 2)                      # MSE
    grad_w = (2 / n) * X.T @ err                  # (2,)  ← X^T (ŷ − y)
    grad_b = (2 / n) * err.sum()
    w -= eta * grad_w; b -= eta * grad_b          # تحديث
    if ep in (1, 2, 5, 10, 20, 50, epochs):
        print(f"epoch {{ep:3d}}: loss={{loss:8.4f}}  w={{w.round(3)}}  b={{b:.3f}}")

# الحل التحليلي (المربعات الصغرى) للمقارنة
Xb = np.column_stack([X, np.ones(n)])
theta = np.linalg.lstsq(Xb, y, rcond=None)[0]
print("closed form: w =", theta[:2].round(3), " b =", theta[2].round(3))
print("true:        w =", true_w, " b =", true_b)'''


def _controls() -> dict:
    c1, c2 = st.columns(2)
    with c1:
        eta = st.select_slider("η", options=[0.01, 0.05, 0.1, 0.3, 0.6, 1.0, 1.2], value=0.1, key="ctrl_linreg_eta")
    with c2:
        epochs = st.slider("epochs", 10, 200, 100, 10, key="ctrl_linreg_epochs")
    return {"eta": eta, "epochs": epochs}


def render() -> None:
    lesson_header(LESSON)
    h2("النموذج", "The model")
    equation(r"\hat{y}_i = \sum_{j=1}^{d} w_j x_{ij} + b \qquad\Longleftrightarrow\qquad \hat{\mathbf{y}} = X\mathbf{w} + b",
             [("x_{ij}", "الخاصية $j$ للملاحظة $i$."), (r"\mathbf{w}", "متجه أوزان بطول $d$ — معامل لكل خاصية."), ("b", "الانحياز (التقاطع)."), (r"X\mathbf{w}", "ضرب مصفوفة `(n, d)` في متجه `(d,)` = تنبؤ لكل ملاحظة `(n,)`.")],
             meaning_ar="الانحدار الخطي في الاقتصاد القياسي، وهو نفسه طبقة كثيفة بوحدة واحدة بلا تنشيط.",
             example_ar="سعر عقار ≈ 900·المساحة + 4000·الغرف − 650·العمر + ثابت.",
             dl_link_ar="`Dense(1)` بلا تنشيط، مدرَّبة بـ MSE، هي انحدار خطي. الشبكة العميقة تضع طبقات لاخطية قبل هذه الطبقة الأخيرة.", title_ar="الانحدار الخطي")
    h2("التدرج", "The gradient")
    equation(r"L = \frac{1}{n}\|X\mathbf{w} + b - \mathbf{y}\|^2, \qquad \frac{\partial L}{\partial \mathbf{w}} = \frac{2}{n}X^{\mathsf T}(\hat{\mathbf{y}} - \mathbf{y}), \qquad \frac{\partial L}{\partial b} = \frac{2}{n}\sum_i(\hat{y}_i - y_i)",
             [(r"\hat{\mathbf{y}} - \mathbf{y}", "متجه الأخطاء `(n,)`."), (r"X^{\mathsf T}(\cdot)", "المنقولة `(d, n)` في الأخطاء `(n,)` = `(d,)`: مساهمة كل خاصية في الخطأ."), (r"\frac{2}{n}", "من مشتقة المربع ومن المتوسط.")],
             meaning_ar="تدرج الوزن $j$ هو متوسط (الخطأ × الخاصية $j$): إن كان الخطأ موجبًا حيث الخاصية موجبة، فالوزن كبير جدًا.",
             example_ar="بخاصية واحدة: $\\partial L/\\partial w = \\frac{2}{n}\\sum_i (\\hat{y}_i - y_i)x_i$ — ما رأيته في محاكي الحقبة.",
             dl_link_ar="هذه صيغة تدرج **أي** طبقة خطية: $X^{\\mathsf T}\\delta$ حيث $\\delta$ التدرج القادم من الأعلى. الانتشار الخلفي يعمّمها.", title_ar="تدرج MSE")
    intuition("اقرأ $X^{\\mathsf T}(\\hat{y} - y)$ كاستجواب: لكل خاصية، هل تتزامن قيمها الموجبة مع أخطاء موجبة؟ إن نعم، وزنها مبالغ فيه — قلّله.")
    code_lab(CodeLab(
        key="ml_linreg", title_ar="انحدار خطي من الصفر مقابل الحل التحليلي", code=CODE, template=True, defaults={"eta": 0.1, "epochs": 100},
        before=Before(goal_ar="تدريب انحدار خطي بخاصيتين بالانحدار التدريجي الكامل، ومقارنة المعلمات المتعلَّمة بالحقيقية وبالحل التحليلي.", stage_ar="نموذج ← تدريب ← تقييم.",
                      inputs_ar="`X (200, 2)` موحدة، `y (200,)` من علاقة معروفة + ضوضاء.", expected_ar="خسارة تهبط من ~38 إلى ~0.25 (تباين الضوضاء)، و`w ≈ (3, −2)`، `b ≈ 5` بالطريقتين.",
                      math_ar="$\\mathbf{w} \\leftarrow \\mathbf{w} - \\eta \\frac{2}{n}X^{\\mathsf T}(\\hat{\\mathbf{y}} - \\mathbf{y})$."),
        explain=[("3-6", "بيانات اصطناعية بمعلمات حقيقية معروفة — أفضل طريقة لاختبار خوارزمية تدريب."),
                 ("8-9", "تهيئة بأصفار (مقبولة للنموذج الخطي؛ للشبكات نحتاج عشوائية كما سنرى)."),
                 ("10-16", "حلقة التدريب: أمامي ← خطأ ← خسارة ← تدرج ← تحديث. دفعة واحدة = كل البيانات (Batch GD)."),
                 ("13", "`X.T @ err`: `(2,200)@(200,)` → `(2,)`. الصيغة المصفوفية للتدرج."),
                 ("21-24", "المربعات الصغرى تحل المسألة جبريًا في خطوة؛ ممكن هنا لأن النموذج خطي والخسارة تربيعية — مستحيل للشبكات العميقة، لذلك نحتاج التدرج.")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- الخسارة لا تصل إلى الصفر بل إلى ≈ 0.25 = تباين الضوضاء (0.5²): الحد الأدنى الممكن. خسارة أقل من الضوضاء تعني حفظًا.\n- جرّب `η = 1.2`: الخاصيتان موحدتان فالانحناء ≈ 2؛ خطوة أكبر من 1 تتباعد.",
    ))
    h3("ماذا يرى الاقتصادي القياسي هنا؟", "The econometrician's view")
    research_note("نفس المعادلة $y = X\\beta + \\epsilon$؛ نفس الحل بالمربعات الصغرى؛ لكن الأهداف تختلف: الاقتصاد القياسي يهتم بـ $\\beta$ (تفسير، أخطاء معيارية، سببية بافتراضات)، والتعلم الآلي يهتم بـ $\\hat{y}$ على بيانات جديدة (تنبؤ). الأوزان في شبكة عميقة **لا** تُفسَّر كمرونات.")
    common_mistake("تدريب انحدار خطي بالتدرج على خصائص غير موحدة (دخل بالآلاف وعمر بالعشرات): معدل تعلم يناسب إحداها يفجّر الأخرى. وحّد الخصائص أولًا (رأيت السبب في معمل التدرج).")
    quiz("ml.linreg", [
        Q("$X$ بالشكل `(500, 4)`؛ شكل `X.T @ err`…", ["(500,)", "(4,)", "(4, 500)"], 1, "تدرج لكل وزن.", kind="shape"),
        Q("الخسارة الدنيا الممكنة مع ضوضاء تباينها 0.25…", ["0", "≈ 0.25", "≈ 0.5"], 1, "لا يمكن التنبؤ بالضوضاء."),
        Q("`Dense(1)` بلا تنشيط + MSE هي…", ["انحدار لوجستي", "انحدار خطي", "تصنيف"], 1, "طبقة خطية."),
        Q("لماذا لا نستخدم الحل التحليلي للشبكات العميقة؟", ["بطيء", "غير موجود: النموذج لاخطي والخسارة غير تربيعية", "يحتاج GPU"], 1, "لا حل مغلق."),
    ])
    takeaway("ŷ = Xw + b، تدرج MSE = (2/n)Xᵀ(ŷ − y). نفس الحلقة تدرّب أي شبكة؛ الحل المغلق رفاهية خطية فقط.")
    lesson_footer(LESSON, ["الانحدار الخطي = Dense(1) + MSE.", "Xᵀ·error هو تدرج كل طبقة خطية.", "الخسارة الدنيا = تباين الضوضاء."])
