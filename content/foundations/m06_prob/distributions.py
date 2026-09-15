import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table

LESSON = Lesson(
    id="foundations.prob.distributions",
    title_ar="الاحتمال والمتغير العشوائي والتوزيع والتوقّع والعينة",
    title_en="Probability, Random Variables, Distributions, Expectation & Sampling",
    module="foundations.prob",
    order=2,
    prerequisites=["foundations.prob.descriptive", "foundations.python.randomness_reproducibility"],
    objectives_ar=[
        "قراءة $P(A)$ و$P(A \\mid B)$ كتكرارات نسبية وشرط.",
        "التمييز بين توزيع برنولي والفئوي والطبيعي، وأين يظهر كل منها في الشبكة.",
        "فهم التوقّع كمتوسط موزون، والعينة كمشاهدات من التوزيع، والضوضاء كجزء لا يُتنبأ به.",
    ],
    terms=["observation", "target"],
    difficulty="beginner",
    summary_ar="الاحتمال تكرار على المدى الطويل؛ برنولي للثنائي، الفئوي للمتعدد، الطبيعي للمتصل؛ التوقّع متوسط موزون.",
)

CODE = '''import numpy as np
rng = np.random.default_rng(0)

# برنولي: نجاح/فشل باحتمال p — مثل "هل تعثر؟"
p = 0.3
draws = rng.uniform(size=10_000) < p
print("Bernoulli(0.3): empirical mean =", draws.mean().round(3), " expectation = p =", p)

# فئوي: k فئات باحتمالات — مخرج Softmax
probs = np.array([0.6, 0.3, 0.1])
cls = rng.choice(3, size=10_000, p=probs)
print("Categorical: empirical =", np.bincount(cls) / 10_000)

# طبيعي: متصل حول متوسط بانحراف — الضوضاء وتهيئة الأوزان
z = rng.normal(loc=50, scale=10, size=10_000)
print("Normal(50, 10): mean =", z.mean().round(2), " std =", z.std().round(2))
print("share within ±1σ:", ((z > 40) & (z < 60)).mean().round(3), "(theory ≈ 0.683)")

# التوقّع كمتوسط موزون: E[X] = Σ x·P(x)
values = np.array([0, 10, 100]); pr = np.array([0.7, 0.25, 0.05])
print("E[X] =", (values * pr).sum())

# الاحتمال الشرطي من جدول: P(default | unemployed)
employment = rng.choice(["employed", "unemployed"], size=10_000, p=[0.85, 0.15])
default = np.where(employment == "unemployed", rng.uniform(size=10_000) < 0.35, rng.uniform(size=10_000) < 0.08)
print("P(default) =", default.mean().round(3))
print("P(default | unemployed) =", default[employment == "unemployed"].mean().round(3))
print("P(default | employed)   =", default[employment == "employed"].mean().round(3))'''


def render() -> None:
    lesson_header(LESSON)
    h2("الاحتمال", "Probability")
    definition("**الاحتمال** $P(A)$ عدد بين 0 و1 يقيس إمكانية وقوع الحدث $A$؛ عمليًا هو التكرار النسبي على المدى الطويل. **الاحتمال الشرطي** $P(A \\mid B)$ احتمال $A$ **علمًا بأن** $B$ وقع: نحصر النظر في الحالات التي فيها $B$.")
    intuition("«احتمال التعثر 8%» يعني من كل 100 عميل يتعثر 8 تقريبًا. «احتمال التعثر بشرط البطالة 35%» يعني من كل 100 **عاطل** يتعثر 35. النموذج التصنيفي يتعلم $P(y \\mid x)$: احتمال الهدف بشرط الخصائص.")
    h2("المتغير العشوائي والتوزيع", "Random variable & distribution")
    definition("**المتغير العشوائي** كمية قيمتها تتحدد بالصدفة (نتيجة التجربة). **التوزيع** يصف احتمال كل قيمة ممكنة. **العينة** `Sample` مجموعة مشاهدات مسحوبة من التوزيع. **الضوضاء** `Noise` الجزء العشوائي الذي لا تفسره الخصائص.")
    table(["التوزيع", "القيم", "المعلمات", "أين في التعلم العميق"],
          [("برنولي", "0 أو 1", "p", "الهدف الثنائي؛ مخرج Sigmoid = p"), ("الفئوي", "1..k", "p₁..pₖ (مجموعها 1)", "الهدف المتعدد؛ مخرج Softmax"),
           ("الطبيعي (غاوس)", "متصل", "μ, σ", "الضوضاء في الانحدار، تهيئة الأوزان، MSE = افتراض ضوضاء طبيعية"), ("المنتظم", "متصل في [a, b]", "a, b", "تهيئة الأوزان، الخلط")],
          ["rtl", "rtl", "code", "rtl"])
    h3("شاهد التوزيعات", "See them")
    rng = np.random.default_rng(1)
    c1, c2 = st.columns(2)
    with c1:
        p = st.slider("برنولي p", 0.0, 1.0, 0.3, 0.05, key="dist_p")
        fig = go.Figure(go.Bar(x=["0", "1"], y=[1 - p, p], marker_color=["#B9B2A6", "#2F6FB5"]))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[0, 1]), title="Bernoulli", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9")
        st.plotly_chart(fig, width="stretch", key="dist_bern")
    with c2:
        mu = st.slider("الطبيعي μ", -3.0, 3.0, 0.0, 0.5, key="dist_mu"); sig = st.slider("σ", 0.3, 3.0, 1.0, 0.1, key="dist_sig")
        xs = np.linspace(-8, 8, 300); pdf = np.exp(-0.5 * ((xs - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
        fig = go.Figure(go.Scatter(x=xs, y=pdf, fill="tozeroy", line=dict(color="#7C5CBF")))
        fig.add_trace(go.Histogram(x=rng.normal(mu, sig, 2000), histnorm="probability density", opacity=0.35, marker_color="#2F6FB5", name="عينة 2000"))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10), title="Normal(μ, σ) + sample", showlegend=False, plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9")
        st.plotly_chart(fig, width="stretch", key="dist_norm")
    h2("التوقّع", "Expectation")
    equation(r"\mathbb{E}[X] = \sum_x x\,P(X = x) \qquad (\text{للمتصل: } \int x\,p(x)\,dx)",
             [(r"\mathbb{E}[X]", "القيمة المتوقعة: المتوسط على المدى الطويل."), ("x P(X=x)", "كل قيمة موزونة باحتمالها.")],
             meaning_ar="متوسط موزون بالاحتمالات. المتوسط الحسابي للعينة يقرّبه.",
             example_ar="مكافأة 0 باحتمال 0.7، 10 باحتمال 0.25، 100 باحتمال 0.05: $\\mathbb{E} = 0 + 2.5 + 5 = 7.5$.",
             dl_link_ar="الخسارة الحقيقية هي توقّع على توزيع البيانات؛ خسارة الدفعة تقدير له بعينة. لهذا تتذبذب خسارة الدفعات، ولهذا يكفي متوسط عدة دفعات.", title_ar="التوقّع")
    code_lab(CodeLab(
        key="prob_dist", title_ar="عينات من التوزيعات الثلاثة + توقّع + احتمال شرطي", code=CODE,
        before=Before(goal_ar="محاكاة التوزيعات، التحقق من أن متوسط العينة يقرّب التوقّع، وحساب احتمال شرطي من بيانات مولّدة.", stage_ar="احتمال.",
                      inputs_ar="بذرة ومعلمات توزيع.", expected_ar="تكرارات تجريبية قريبة من النظرية، E[X] = 7.5، وP(default | unemployed) ≈ 0.35 مقابل P(default | employed) ≈ 0.08."),
        explain=[("4-7", "برنولي بالمحاكاة: المتوسط التجريبي ≈ p."), ("9-12", "الفئوي: `choice` باحتمالات؛ `bincount` يعدّ."), ("14-17", "الطبيعي: 68% تقريبًا ضمن ±σ."), ("19-21", "التوقّع بالتعريف."),
                 ("23-28", "احتمال شرطي = تصفية على الشرط ثم متوسط. هذا ما يتعلمه المصنف لكن مع شروط كثيرة معًا.")],
        run=run_printed(CODE),
        after_ar="- الفرق بين 0.35 و0.08 هو «المعلومة» التي تجعل `employment` خاصية مفيدة.\n- التكرارات التجريبية ليست مطابقة تمامًا للنظرية: ضوضاء العينة تتناقص مع حجمها.",
    ))
    common_mistake("«الاحتمال 0.9 من النموذج يعني 90% من الحالات المشابهة صحيحة». فقط إذا كان النموذج **معايَرًا** `calibrated`. الشبكات كثيرًا ما تكون واثقة أكثر من اللازم؛ الاحتمال المخرج ترتيب أكثر منه تكرارًا مضمونًا.")
    quiz("prob.dist", [
        Q("$P(A \\mid B)$ تعني…", ["احتمال A وB معًا", "احتمال A علمًا بوقوع B", "احتمال B علمًا بوقوع A"], 1, "الشرط يحصر الحالات."),
        Q("مخرج Sigmoid يمثل معلمة توزيع…", ["الطبيعي", "برنولي", "الفئوي"], 1, "p للثنائي."),
        Q("$\\mathbb{E}[X]$ لقيم 1 (0.5) و3 (0.5)…", ["2", "4", "1.5"], 0, "0.5 + 1.5.", kind="equation"),
        Q("خسارة الدفعة تتذبذب بين الدفعات لأنها…", ["خاطئة", "تقدير بعينة لتوقّع", "تعتمد على معدل التعلم"], 1, "ضوضاء العينة."),
    ])
    takeaway("الاحتمال تكرار؛ الشرطي يحصر الحالات؛ برنولي/فئوي/طبيعي تظهر في المخرجات والضوضاء والتهيئة؛ التوقّع متوسط موزون تقرّبه العينة.")
    lesson_footer(LESSON, ["P(y | x) هو ما يتعلمه المصنف.", "Sigmoid ↔ برنولي، Softmax ↔ فئوي، MSE ↔ طبيعي.", "خسارة الدفعة تقدير عيني للتوقّع."])
