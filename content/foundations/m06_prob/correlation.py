import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, research_note, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.prob.correlation",
    title_ar="التغاير والارتباط",
    title_en="Covariance & Correlation",
    module="foundations.prob",
    order=3,
    prerequisites=["foundations.prob.descriptive"],
    objectives_ar=[
        "حساب التغاير والارتباط وتفسير الإشارة والحجم.",
        "معرفة أن الارتباط يقيس العلاقة الخطية فقط، وأنه ليس سببية.",
        "استخدام مصفوفة الارتباط لفحص الخصائص المتكررة قبل النموذج.",
    ],
    terms=["feature"],
    difficulty="beginner",
    summary_ar="الارتباط تغاير موحّد بين −1 و1؛ يقيس الخطية فقط ولا يثبت السببية.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(0)
n = 500
income = rng.normal(5000, 1500, n)
debt = 0.4 * income + rng.normal(0, 800, n)         # مرتبط موجبًا بالدخل
age = rng.normal(40, 10, n)                         # مستقل
late = np.clip(8 - 0.001 * income + rng.normal(0, 1.5, n), 0, None)   # مرتبط سالبًا

def corr(a, b):
    return ((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std())

print("cov(income, debt) =", np.cov(income, debt, bias=True)[0, 1].round(1))
print("corr(income, debt)=", corr(income, debt).round(3))
print("corr(income, age) =", corr(income, age).round(3))
print("corr(income, late)=", corr(income, late).round(3))

# مصفوفة الارتباط بين كل الأزواج
X = np.column_stack([income, debt, age, late])
print(np.corrcoef(X, rowvar=False).round(2))

# علاقة قوية لكن غير خطية → ارتباط قريب من الصفر
u = rng.uniform(-3, 3, n); v = u ** 2 + rng.normal(0, 0.5, n)
print("corr(u, u^2) =", corr(u, v).round(3), " ← علاقة تامة تقريبًا لكن الارتباط الخطي لا يراها")'''


def render() -> None:
    lesson_header(LESSON)
    h2("التغاير والارتباط", "Covariance & correlation")
    equation(r"\text{cov}(x, y) = \frac{1}{n}\sum_i (x_i - \bar{x})(y_i - \bar{y}), \qquad r = \frac{\text{cov}(x, y)}{\sigma_x\,\sigma_y} \in [-1, 1]",
             [(r"\text{cov}", "التغاير: هل يميل $x$ و$y$ إلى الابتعاد عن متوسطيهما في نفس الاتجاه (موجب) أم عكسه (سالب)؟ وحدته حاصل ضرب الوحدتين."), ("r", "الارتباط (بيرسون): التغاير مقسومًا على الانحرافين فيصبح بلا وحدة بين −1 و1.")],
             meaning_ar="$r$ قريب من 1: علاقة خطية طردية قوية؛ قريب من −1: عكسية قوية؛ قريب من 0: لا علاقة **خطية**.",
             example_ar="الدخل والدين: $r \\approx 0.6$؛ الدخل والعمر: $r \\approx 0$.",
             dl_link_ar="مصفوفة الارتباط تكشف خصائص شبه متطابقة (تكرار) وخصائص مرتبطة بالهدف بقوة مشبوهة (تسريب محتمل).", title_ar="التغاير والارتباط")
    rng = np.random.default_rng(3)
    r_target = st.slider("الارتباط المستهدف r", -1.0, 1.0, 0.7, 0.1, key="corr_r")
    a = rng.normal(size=300); b = r_target * a + np.sqrt(max(0.0, 1 - r_target ** 2)) * rng.normal(size=300)
    fig = go.Figure(go.Scatter(x=a, y=b, mode="markers", marker=dict(color="#2F6FB5", size=7, opacity=0.7)))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title=f"r (measured) = {np.corrcoef(a, b)[0, 1]:.2f}", xaxis_title="x", yaxis_title="y", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9")
    st.plotly_chart(fig, width="stretch", key="corr_scatter")
    code_lab(CodeLab(
        key="prob_corr", title_ar="ارتباطات من بيانات مولّدة + مصفوفة ارتباط + فخ اللاخطية", code=CODE,
        before=Before(goal_ar="حساب الارتباط يدويًا وبـ NumPy، بناء مصفوفة ارتباط، ورؤية علاقة قوية يفشل الارتباط الخطي في رؤيتها.", stage_ar="إحصاء ← فحص الخصائص.",
                      inputs_ar="4 خصائص مولّدة بعلاقات معروفة.", expected_ar="ارتباط موجب للدين، ~0 للعمر، سالب للتأخيرات، مصفوفة 4×4، وارتباط ~0 لعلاقة u² رغم قوتها."),
        explain=[("4-7", "نبني علاقات معروفة لنتحقق أن الارتباط يكتشفها."), ("9-10", "الصيغة حرفيًا: متوسط حاصل ضرب الانحرافات مقسومًا على الانحرافين."), ("12-15", "إشارات وأحجام متوقعة."),
                 ("18-19", "`corrcoef` لكل الأزواج؛ القطر = 1 دائمًا. ابحث عن قيم قريبة من ±1 خارج القطر (تكرار)."), ("22-23", "الفخ: $v = u^2$ علاقة تامة لكنها متناظرة فيلغي الجانبان بعضهما.")],
        run=run_printed(CODE),
        after_ar="- الشبكات العصبية تلتقط العلاقات اللاخطية التي يتجاهلها الارتباط؛ لذلك «ارتباط ضعيف» لا يعني «خاصية عديمة الفائدة».\n- ارتباط مرتفع جدًا بين خاصيتين (> 0.95) يقترح حذف إحداهما أو على الأقل الانتباه للتفسير.",
    ))
    research_note("**الارتباط ليس سببية.** مبيعات المثلجات وحوادث الغرق مرتبطتان (الصيف). في الاقتصاد: المتغيرات الكامنة، السببية العكسية، والاتجاه الزمني المشترك كلها تولّد ارتباطًا زائفًا. النموذج التنبؤي يستغل الارتباط ولا يحتاج السببية — لكن **تفسيرك** يحتاجها.")
    common_mistake("حذف خاصية لأن ارتباطها بالهدف ضعيف: العلاقة قد تكون لاخطية أو تفاعلية (تظهر فقط مع خاصية أخرى) — وهذا بالضبط ما تجيد الشبكات التقاطه.")
    quiz("prob.corr", [
        Q("$r = -0.9$ يعني…", ["لا علاقة", "علاقة خطية عكسية قوية", "y يسبب x"], 1, "الإشارة اتجاه، الحجم قوة."),
        Q("$y = x^2$ على مجال متناظر: الارتباط الخطي ≈", ["1", "0", "−1"], 1, "الخطي لا يرى التناظر."),
        Q("ارتباط 0.99 بين خاصية والهدف في بيانات اقتصادية…", ["ممتاز، استخدمها", "مشبوه: تسريب محتمل", "لا يعني شيئًا"], 1, "افحص هل تُعرف بعد الحدث."),
    ])
    takeaway("r = cov/(σσ) بين −1 و1؛ خطي فقط؛ ليس سببية. مصفوفة الارتباط تكشف التكرار والتسريب.")
    lesson_footer(LESSON, ["التغاير له وحدة؛ الارتباط بلا وحدة.", "علاقة لاخطية قوية قد تعطي r ≈ 0.", "ارتباط ≠ سببية."])
