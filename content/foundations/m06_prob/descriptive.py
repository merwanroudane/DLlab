import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from labs.datasets import loan_default

LESSON = Lesson(
    id="foundations.prob.descriptive",
    title_ar="المتوسط والوسيط والتباين والانحراف المعياري",
    title_en="Mean, Median, Variance & Standard Deviation",
    module="foundations.prob",
    order=1,
    prerequisites=["foundations.math.summation_average_abs"],
    objectives_ar=[
        "حساب المقاييس الأربعة وفهم ما يقيسه كل منها.",
        "فهم لماذا يُستخدم المتوسط والانحراف في التوحيد القياسي `Standardization`.",
        "معرفة متى يضلّل المتوسط (قيم شاذة) ويفيد الوسيط.",
    ],
    terms=["feature"],
    difficulty="beginner",
    summary_ar="المتوسط مركز، التباين انتشار، الانحراف جذره بنفس الوحدة؛ التوحيد = (x − μ)/σ.",
)

CODE = '''import numpy as np
x = np.array([4200, 6100, 2900, 8800, 5100, 3300, 45000.])   # دخل 7 عملاء (الأخير شاذ)

mu = x.mean()
med = np.median(x)
var = ((x - mu) ** 2).mean()          # التباين: متوسط مربع البعد عن المتوسط
std = np.sqrt(var)                    # الانحراف المعياري
print(f"mean={mu:.1f} median={med:.1f} var={var:.1f} std={std:.1f}")
print("np.var/std:", x.var().round(1), x.std().round(1))

z = (x - mu) / std                    # التوحيد القياسي: متوسط 0، انحراف 1
print("z-scores:", z.round(2))
print("z mean≈", z.mean().round(6), " z std≈", z.std().round(6))

x2 = x[:-1]                           # بلا القيمة الشاذة
print(f"without outlier: mean={x2.mean():.1f} median={np.median(x2):.1f} std={x2.std():.1f}")'''


def render() -> None:
    lesson_header(LESSON)
    h2("أربعة أرقام تلخص عمودًا", "Four numbers that summarize a column")
    equation(r"\mu = \frac{1}{n}\sum_{i=1}^{n} x_i, \qquad \sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2, \qquad \sigma = \sqrt{\sigma^2}",
             [(r"\mu", "المتوسط: مركز الثقل."), (r"\sigma^2", "التباين: متوسط مربع البعد عن المتوسط — يقيس الانتشار بوحدة مربعة."), (r"\sigma", "الانحراف المعياري: الجذر ليعود إلى وحدة البيانات.")],
             meaning_ar="أين تتمركز القيم وكم تنتشر حولها.",
             example_ar="$\\{2, 4, 6\\}$: $\\mu = 4$، $\\sigma^2 = (4 + 0 + 4)/3 = 2.67$، $\\sigma = 1.63$.",
             dl_link_ar="التوحيد القياسي لكل خاصية: $z = (x - \\mu)/\\sigma$ يجعل المتوسط 0 والانحراف 1 فلا تهيمن خاصية كبيرة المقياس على التدرج. يُحسب $\\mu, \\sigma$ من **بيانات التدريب فقط**.", title_ar="المتوسط والتباين والانحراف")
    definition("**الوسيط** `Median` القيمة الوسطى بعد الترتيب؛ نصف القيم أقل منه ونصفها أكبر. لا يتأثر بالقيم الشاذة كما يتأثر المتوسط.")
    intuition("المتوسط «مركز الثقل»: قيمة شاذة واحدة بعيدة تسحبه بقوة. الوسيط «الشخص في المنتصف»: لا يبالي كم يبعد الأغنى.")
    df = loan_default()
    inc = df["income"].to_numpy()
    fig = go.Figure(go.Histogram(x=inc, nbinsx=40, marker_color="#2F6FB5", opacity=0.8))
    fig.add_vline(x=inc.mean(), line=dict(color="#C8473A", width=2), annotation_text=f"mean {inc.mean():.0f}")
    fig.add_vline(x=np.median(inc), line=dict(color="#2E8B57", width=2, dash="dot"), annotation_text=f"median {np.median(inc):.0f}")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="income", yaxis_title="count", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9")
    st.plotly_chart(fig, width="stretch", key="desc_hist")
    st.caption("توزيع الدخل في بيانات القروض: ذيل يمين طويل يسحب المتوسط فوق الوسيط.")
    code_lab(CodeLab(
        key="prob_desc", title_ar="المقاييس الأربعة والتوحيد القياسي وأثر القيمة الشاذة", code=CODE,
        before=Before(goal_ar="حساب المقاييس يدويًا وبـ NumPy، توحيد عمود، ورؤية أثر قيمة شاذة واحدة.", stage_ar="إحصاء ← إعداد البيانات.",
                      inputs_ar="7 قيم دخل إحداها شاذة.", expected_ar="متوسط أكبر بكثير من الوسيط، انحراف ضخم، z-scores، ثم مقاييس معقولة بلا الشاذة."),
        explain=[("4-9", "المقاييس بالصيغ ثم بدوال NumPy المطابقة."), ("11-13", "التوحيد: طرح المتوسط والقسمة على الانحراف. النتيجة دائمًا متوسط 0 وانحراف 1."), ("15-16", "إزالة الشاذة: المتوسط يهبط من ~10.8 ألف إلى ~5 آلاف، والوسيط يكاد لا يتغير.")],
        run=run_printed(CODE),
        after_ar="- القيمة الشاذة 45000 لها z-score ≈ 2.4 بينما بقية القيم قريبة من الصفر — التوحيد يكشف الشواذ.\n- عند التحجيم قبل التدريب، قيمة شاذة واحدة تضغط بقية القيم إلى نطاق ضيق؛ لهذا نعالج الشواذ أولًا.",
    ))
    why("لماذا نحتاج التوحيد في الشبكات؟ خاصية بمقياس آلاف (الدخل) وأخرى بعشرات (العمر) تعطيان تدرجات بمقاييس مختلفة جدًا فيتعرج الانحدار (رأيت هذا في معمل التدرج مع الوعاء الإهليلجي).")
    common_mistake("حساب $\\mu, \\sigma$ من كل البيانات ثم التقسيم: بيانات الاختبار تسرّبت إلى التحجيم. الترتيب الصحيح: قسّم، ثم احسب من التدريب، ثم طبّق على الباقي.")
    quiz("prob.desc", [
        Q("$\\{1, 2, 3, 4, 100\\}$: أيهما أكبر؟", ["المتوسط", "الوسيط", "متساويان"], 0, "الشاذة تسحب المتوسط.", kind="equation"),
        Q("بعد التوحيد القياسي يصبح متوسط الخاصية…", ["1", "0", "لا يتغير"], 1, "z = (x−μ)/σ."),
        Q("من أين تُحسب μ وσ للتحجيم؟", ["من كل البيانات", "من بيانات التدريب فقط", "من الاختبار"], 1, "منع التسريب."),
    ])
    takeaway("μ مركز، σ² انتشار، σ بوحدة البيانات، الوسيط مقاوم للشواذ. التوحيد (x−μ)/σ من التدريب فقط.")
    lesson_footer(LESSON, ["أربعة مقاييس ومعناها.", "z-score يوحّد المقاييس ويكشف الشواذ.", "احسب من التدريب، طبّق على الباقي."])
