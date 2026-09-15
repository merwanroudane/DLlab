import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation, worked_steps
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.math.summation_average_abs",
    title_ar="رمز المجموع Σ والمتوسط والقيمة المطلقة",
    title_en="Summation Σ, Average & Absolute Value",
    module="foundations.math",
    order=4,
    prerequisites=["foundations.math.notation"],
    objectives_ar=[
        "قراءة $\\sum_{i=1}^{n}$ وترجمته إلى حلقة `for` أو `np.sum`.",
        "فهم المتوسط كمجموع مقسوم على العدد، ولماذا تظهر $\\frac{1}{n}$ في كل خسارة.",
        "فهم القيمة المطلقة $|x|$ ودورها في MAE مقابل المربع في MSE.",
    ],
    terms=["loss"],
    difficulty="beginner",
    summary_ar="Σ حلقة جمع، المتوسط Σ/n، |x| مسافة بلا إشارة. MSE وMAE مبنيتان عليها.",
)

CODE = '''import numpy as np

y_true = np.array([52., 55., 61., 64.])
y_pred = np.array([50., 58., 60., 70.])
errors = y_pred - y_true                  # e_i
n = len(errors)

# Σ_{i=1}^{n} e_i  بثلاث طرق
total = 0
for e in errors:
    total += e
print(total, errors.sum(), np.sum(errors))

print("mean  :", errors.sum() / n, errors.mean())            # (1/n) Σ e_i
print("|e|   :", np.abs(errors))                             # القيمة المطلقة
print("MAE   :", np.abs(errors).mean())                      # (1/n) Σ |e_i|
print("MSE   :", (errors ** 2).mean())                       # (1/n) Σ e_i²
print("RMSE  :", np.sqrt((errors ** 2).mean()))              # بنفس وحدة y

# مجموع مزدوج Σ_i Σ_j  = مجموع كل عناصر مصفوفة
M = np.array([[1, 2], [3, 4]])
print(M.sum(), M.sum(axis=0), M.sum(axis=1))'''


def render() -> None:
    lesson_header(LESSON)
    h2("رمز المجموع", "Sigma notation")
    equation(r"\sum_{i=1}^{n} x_i = x_1 + x_2 + \cdots + x_n",
             [(r"\sum", "«اجمع»."), ("i = 1", "ابدأ العدّاد $i$ من 1."), ("n", "انتهِ عند $n$."), ("x_i", "الحد الذي يُجمع في كل خطوة.")],
             meaning_ar="حلقة جمع مضغوطة في رمز واحد.",
             example_ar="$\\sum_{i=1}^{3} i^2 = 1 + 4 + 9 = 14$.",
             dl_link_ar="كل خسارة تبدأ بـ $\\sum$ على الملاحظات؛ الطبقة الكثيفة $\\sum_j w_j x_j$؛ Softmax تقسم على $\\sum_k e^{z_k}$.", title_ar="المجموع")
    intuition("اقرأ $\\sum_{i=1}^{n} x_i$ حرفيًا كـ `total = 0; for i in range(1, n+1): total += x[i]`. لا شيء أكثر.")
    h2("المتوسط", "Average / mean")
    equation(r"\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i", [(r"\bar{x}", "المتوسط (الشرطة فوق الرمز)."), (r"\frac{1}{n}", "القسمة على العدد كي لا يعتمد الناتج على حجم البيانات.")],
             meaning_ar="مجموع القيم مقسومًا على عددها.",
             example_ar="$\\{2, 4, 9\\}$: $(2+4+9)/3 = 5$.",
             dl_link_ar="لهذا تبدأ MSE وMAE والإنتروبيا المتقاطعة بـ $\\frac{1}{n}$: خسارة لا تعتمد على حجم الدفعة.", title_ar="المتوسط")
    h2("القيمة المطلقة", "Absolute value")
    equation(r"|x| = \begin{cases} x & x \ge 0 \\ -x & x < 0 \end{cases}", [("|x|", "المسافة عن الصفر بلا إشارة.")],
             meaning_ar="تحوّل السالب إلى موجب وتترك الموجب.", example_ar="$|-3| = 3$، $|3| = 3$.",
             dl_link_ar="MAE تستخدم $|e_i|$؛ MSE تستخدم $e_i^2$. كلاهما يلغي الإشارة، لكن المربع يضخّم الأخطاء الكبيرة.", title_ar="القيمة المطلقة")
    worked_steps([
        ("الأخطاء $e_i = \\hat{y}_i - y_i$", r"e = (-2,\ 3,\ -1,\ 6)"),
        ("القيم المطلقة", r"|e| = (2,\ 3,\ 1,\ 6) \Rightarrow \text{MAE} = 12/4 = 3"),
        ("المربعات", r"e^2 = (4,\ 9,\ 1,\ 36) \Rightarrow \text{MSE} = 50/4 = 12.5"),
        ("الجذر لإرجاع الوحدة", r"\text{RMSE} = \sqrt{12.5} \approx 3.54"),
    ])
    table(["المقياس", "الصيغة", "حساس للقيم الشاذة؟", "الوحدة"],
          [("MAE", "(1/n) Σ |e_i|", "أقل", "نفس y"), ("MSE", "(1/n) Σ e_i²", "أكثر (المربع يضخّم)", "y²"), ("RMSE", "√MSE", "أكثر", "نفس y")],
          ["ltr", "code", "rtl", "rtl"])
    code_lab(CodeLab(
        key="math_sigma", title_ar="Σ والمتوسط و|x| في NumPy", code=CODE,
        before=Before(goal_ar="ترجمة الرموز الثلاثة إلى كود وحساب MAE وMSE وRMSE يدويًا.", stage_ar="رياضيات ← الخسارة.",
                      inputs_ar="أربع قيم حقيقية وأربعة تنبؤات.", expected_ar="نفس المجموع بثلاث طرق، ثم MAE=3، MSE=12.5، RMSE≈3.54."),
        explain=[("8-12", "Σ بحلقة، ثم بـ `sum()`، ثم `np.sum` — الثلاثة متطابقة."),
                 ("14", "المتوسط = المجموع / n = `mean()`."),
                 ("15-18", "القيمة المطلقة ثم MAE؛ المربع ثم MSE؛ الجذر ثم RMSE. لاحظ أن الخطأ 6 ساهم بـ 36 في MSE لكن 6 فقط في MAE."),
                 ("21-22", "المجموع على مصفوفة: كل العناصر، أو لكل عمود، أو لكل صف — نفس `axis` من قبل.")],
        run=run_printed(CODE),
        after_ar="- مجموع الأخطاء الخام 6 لا يفيد (السالب يلغي الموجب) — لهذا نأخذ المطلق أو المربع.\n- MSE = 12.5 مقابل MAE = 3: الخطأ الكبير (6) يهيمن على MSE.",
    ))
    common_mistake("نسيان $\\frac{1}{n}$ فيصبح مجموع الخسارة يكبر مع حجم الدفعة، فيبدو أن الدفعات الكبيرة «أسوأ». الخسارة تُقارَن كمتوسط.")
    quiz("math.sigma", [
        Q("$\\sum_{i=1}^{4} 2i$ يساوي…", ["8", "20", "10"], 1, "2+4+6+8.", kind="equation"),
        Q("أخطاء (−1, 1): مجموعها ومتوسط مطلقها؟", ["0 و 1", "0 و 0", "2 و 1"], 0, "الإشارات تتلاشى في المجموع الخام."),
        Q("لماذا MSE أكثر حساسية للقيم الشاذة من MAE؟", ["بسبب القسمة على n", "لأن المربع يضخّم الأخطاء الكبيرة", "لأنها أسرع"], 1, "36 مقابل 6."),
    ])
    takeaway("Σ حلقة، 1/n متوسط، |x| مسافة. MAE بالمطلق، MSE بالمربع، RMSE بالجذر.")
    lesson_footer(LESSON, ["Σ = for loop = np.sum.", "1/n يجعل الخسارة مستقلة عن حجم الدفعة.", "المربع يضخّم الشواذ؛ المطلق لا."])
