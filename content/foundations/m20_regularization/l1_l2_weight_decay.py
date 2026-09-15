import numpy as np
import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto

LESSON = Lesson(
    id="foundations.regularization.l1_l2_weight_decay",
    title_ar="L1 وL2 وتضاؤل الأوزان",
    title_en="L1, L2 & Weight Decay",
    module="foundations.regularization",
    order=1,
    prerequisites=["foundations.generalization.under_overfitting", "foundations.calculus.derivative"],
    objectives_ar=["إضافة حد عقوبة إلى الهدف واشتقاق أثره على التدرج.", "L2 = تضاؤل الأوزان (تقليص نسبي)؛ L1 = تصفير (اختيار خصائص).", "ضبط λ ورؤية أثره على الأوزان والفجوة."],
    terms=["norm", "loss"],
    labs=["labs.regularization_lab"],
    difficulty="intermediate",
    summary_ar="الهدف = الخسارة + λ‖w‖²: كل تحديث يقلّص الأوزان بنسبة (1 − 2ηλ). L1 يصفّر؛ L2 يصغّر. λ يوازن الملاءمة والبساطة.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(0)
n, d = 60, 20
X = rng.normal(size=(n, d)); w_true = np.zeros(d); w_true[:3] = [3.0, -2.0, 1.5]   # 3 خصائص مفيدة فقط
y = X @ w_true + rng.normal(0, 0.5, n)
X_te = rng.normal(size=(200, d)); y_te = X_te @ w_true + rng.normal(0, 0.5, 200)

def fit(lam, kind, epochs=400, eta=0.05):
    w = np.zeros(d)
    for _ in range(epochs):
        g = 2 * X.T @ (X @ w - y) / n
        if kind == "l2": g = g + 2 * lam * w
        if kind == "l1": g = g + lam * np.sign(w)
        w -= eta * g
    return w

for kind, lam in [("none", 0), ("l2", {lam}), ("l1", {lam})]:
    w = fit(lam, kind)
    tr = np.mean((X @ w - y)**2); te = np.mean((X_te @ w - y_te)**2)
    print(f"{{kind:<5}} lam={{lam:<5}} train MSE={{tr:.3f}} test MSE={{te:.3f}} |w|={{np.abs(w).sum():6.2f}} zeros={{(np.abs(w) < 1e-3).sum():>2}}  w[:5]={{w[:5].round(2)}}")'''


def _controls() -> dict:
    return {"lam": st.select_slider("λ", options=[0.001, 0.01, 0.05, 0.1, 0.3, 1.0], value=0.05, key="ctrl_l2_lam")}


def render() -> None:
    lesson_header(LESSON)
    h2("العقوبة", "The penalty")
    equation(r"J(\theta) = L(\theta) + \lambda\,\Omega(\theta), \qquad \Omega_{L2} = \sum_j w_j^2 = \|w\|_2^2, \qquad \Omega_{L1} = \sum_j |w_j| = \|w\|_1",
             [("J", "الهدف الكامل الذي نقلله."), ("L", "خسارة البيانات."), (r"\lambda", "قوة التنظيم (معلمة فائقة): 0 بلا تنظيم، كبيرة = أوزان صغيرة جدًا (قصور)."), (r"\Omega", "حد يعاقب الأوزان الكبيرة. الانحيازات لا تُعاقب عادةً.")],
             meaning_ar="نضيف إلى الخسارة «ضريبة» على حجم الأوزان: النموذج يوازن بين ملاءمة البيانات وإبقاء الأوزان صغيرة.",
             example_ar="w = (3, −2, 0.5): L2 = 13.25، L1 = 5.5.",
             dl_link_ar="`Dense(64, kernel_regularizer=l2(1e-4))` في Keras؛ `weight_decay=` في محسّنات PyTorch (لـ SGD يكافئ L2؛ AdamW يفصله بصورة صحيحة).", title_ar="الهدف المنظَّم")
    equation(r"\text{L2:}\quad \frac{\partial J}{\partial w} = \frac{\partial L}{\partial w} + 2\lambda w \;\Rightarrow\; w \leftarrow (1 - 2\eta\lambda)\,w - \eta\frac{\partial L}{\partial w}",
             [(r"(1 - 2\eta\lambda)", "عامل تقليص أقل من 1 في كل تحديث: **تضاؤل الأوزان** `weight decay`. الأوزان التي لا تدعمها البيانات تنكمش نحو الصفر تدريجيًا.")],
             meaning_ar="L2 لا تصفّر الأوزان بل تصغّرها نسبيًا؛ L1 تدرجها ±λ ثابت فتدفع الأوزان الصغيرة إلى الصفر بالضبط (اختيار خصائص).",
             example_ar="η = 0.1، λ = 0.01: كل تحديث يضرب الأوزان في 0.998 قبل خطوة التدرج.",
             dl_link_ar="لماذا تحسّن التعميم؟ أوزان صغيرة = دالة أنعم = أقل حساسية للضوضاء = تباين أقل.", title_ar="تضاؤل الأوزان")
    intuition("L2 كشد مطاطي يسحب كل وزن نحو الصفر بقوة تتناسب مع حجمه: الكبير يُسحب أكثر لكنه لا يصل. L1 كاحتكاك ثابت: الأوزان الضعيفة تتوقف عند الصفر تمامًا.")
    code_lab(CodeLab(
        key="reg_l1l2", title_ar="بلا تنظيم مقابل L2 مقابل L1 على مسألة بخصائص كثيرة غير مفيدة", code=CODE, template=True, defaults={"lam": 0.05},
        before=Before(goal_ar="60 ملاحظة و20 خاصية (3 مفيدة فقط): مقارنة الخطأ على التدريب والاختبار ومجموع |w| وعدد الأصفار.", stage_ar="التنظيم.",
                      inputs_ar="λ من عنصر التحكم.", expected_ar="بلا تنظيم: أفضل تدريب وأسوأ اختبار؛ L2 يصغّر كل الأوزان ويحسّن الاختبار؛ L1 يصفّر معظم الخصائص غير المفيدة."),
        explain=[("3-6", "خصائص كثيرة نسبة إلى الملاحظات: وضع مثالي لفرط التخصيص."), ("8-15", "الانحدار التدريجي مع حد العقوبة في التدرج: L2 يضيف 2λw، L1 يضيف λ·sign(w)."), ("17-20", "الجدول: التدريب مقابل الاختبار، وحجم الأوزان، وعدد الأصفار.")],
        controls=_controls, run=run_printed(CODE, template=True),
        after_ar="- بلا تنظيم: 17 خاصية ضجيج تحصل على أوزان غير صفرية تلائم الضوضاء.\n- L1 عند λ ≈ 0.05: تبقى الخصائص الثلاث تقريبًا فقط — «اختيار خصائص» مجاني.\n- λ كبيرة (1.0): كل الأوزان تنكمش والتدريب نفسه يسوء (قصور).",
    ))
    compare_table(["", "L2 (Ridge / weight decay)", "L1 (Lasso)"],
                  [("أثر التدرج", "يتناسب مع w (تقليص نسبي)", "ثابت ±λ (دفع إلى الصفر)"), ("الأوزان", "صغيرة، نادرًا صفر", "كثير منها صفر بالضبط"), ("الاستخدام", "الافتراضي في الشبكات", "اختيار خصائص، نماذج مبعثرة"), ("Keras", "l2(λ)", "l1(λ)"), ("قابلية الاشتقاق", "ناعمة", "غير مشتقة عند 0 (يُستخدم sign)")],
                  ["rtl", "rtl", "rtl"])
    research_note("في Adam، `weight_decay` كـ L2 داخل التدرج يتفاعل مع التكيف بصورة غير مقصودة؛ AdamW يطبّق التقليص مباشرة على الأوزان. لذلك AdamW الافتراضي الحديث عندما تريد تضاؤل أوزان مع Adam.")
    st.button("افتح معمل التنظيم", icon=":material/science:", on_click=goto, args=("labs.regularization_lab",), key="l2_lab")
    common_mistake("تطبيق L2 على الانحيازات: يزيح المخرج بلا فائدة تعميمية. الأطر تطبّقه على الأوزان (kernel) افتراضيًا.")
    quiz("reg.l2", [
        Q("L2 تُعرف أيضًا بـ…", ["Dropout", "تضاؤل الأوزان", "الإيقاف المبكر"], 1, "عامل التقليص."),
        Q("أي تنظيم يصفّر الأوزان بالضبط؟", ["L2", "L1", "كلاهما"], 1, "تدرج ثابت."),
        Q("λ كبيرة جدًا تسبب…", ["فرط تخصيص", "قصور تعلم", "تباعدًا"], 1, "أوزان صغيرة جدًا."),
        Q("عامل التقليص في كل تحديث مع L2…", ["1 + 2ηλ", "1 − 2ηλ", "ηλ"], 1, "أقل من 1."),
    ])
    takeaway("J = L + λΩ. L2 يصغّر (تضاؤل)، L1 يصفّر (اختيار). λ يوازن؛ لا عقوبة على الانحيازات؛ AdamW مع Adam.")
    lesson_footer(LESSON, ["العقوبة في الهدف والتدرج.", "(1 − 2ηλ) تضاؤل.", "تجربة بخصائص ضجيج."])
