import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.code_lab import Before, CodeLab, code_lab, run_printed
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.math_explainer import equation, worked_steps
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.prob.likelihood",
    title_ar="الاحتمال الأرجح ولوغاريتمه: من أين تأتي دوال الخسارة",
    title_en="Likelihood & Log-Likelihood: Where Loss Functions Come From",
    module="foundations.prob",
    order=4,
    prerequisites=["foundations.prob.distributions", "foundations.math.exponent_log"],
    objectives_ar=[
        "التمييز بين الاحتمال (بيانات مجهولة، معلمات معلومة) والاحتمال الأرجح (بيانات معلومة، معلمات مجهولة).",
        "اشتقاق خسارة الإنتروبيا المتقاطعة الثنائية من الاحتمال الأرجح لبرنولي.",
        "رؤية أن MSE هي سالب لوغاريتم الاحتمال الأرجح لضوضاء طبيعية.",
    ],
    terms=["loss", "prediction", "target"],
    difficulty="intermediate",
    summary_ar="تدريب النموذج = تعظيم الاحتمال الأرجح = تقليل سالب لوغاريتمه؛ BCE من برنولي وMSE من الطبيعي.",
)

CODE = '''import numpy as np

y = np.array([1, 1, 0, 1, 0, 1, 1, 1])        # 6 نجاحات من 8 مشاهدات
n, k = len(y), y.sum()

def likelihood(p):      # L(p) = p^k (1-p)^(n-k)
    return p**k * (1-p)**(n-k)

def nll(p):             # سالب لوغاريتم الاحتمال الأرجح
    return -(k*np.log(p) + (n-k)*np.log(1-p))

for p in [0.3, 0.5, 0.75, 0.9]:
    print(f"p={p:<5} L={likelihood(p):.6f}  -logL={nll(p):.4f}")
print("best p (analytic) = k/n =", k/n)

# نفس الشيء بصيغة الإنتروبيا المتقاطعة الثنائية لكل مشاهدة:
p_hat = 0.75
bce = -np.mean(y*np.log(p_hat) + (1-y)*np.log(1-p_hat))
print("BCE per observation at p=0.75:", bce.round(4), " = nll/n:", (nll(0.75)/n).round(4))

# انحدار: ضوضاء طبيعية ⇒ سالب اللوغاريتم ∝ مجموع مربعات الخطأ
y_true = np.array([52., 55., 61.]); y_pred = np.array([50., 58., 60.]); sigma = 3.0
nll_gauss = np.sum(0.5*((y_true - y_pred)/sigma)**2 + np.log(sigma*np.sqrt(2*np.pi)))
print("Gaussian NLL:", nll_gauss.round(4), "  SSE/(2σ²):", (((y_true-y_pred)**2).sum()/(2*sigma**2)).round(4), "+ const")'''


def render() -> None:
    lesson_header(LESSON)
    h2("احتمال أم احتمال أرجح؟", "Probability vs likelihood")
    definition("**الاحتمال** $P(\\text{data} \\mid \\theta)$: المعلمات $\\theta$ معلومة، نسأل عن البيانات. **الاحتمال الأرجح** `Likelihood` $\\mathcal{L}(\\theta) = P(\\text{data} \\mid \\theta)$ **نفس التعبير** لكن البيانات معلومة ونقرأه كدالة في $\\theta$: أي قيمة للمعلمات تجعل ما شاهدناه أكثر ترجيحًا؟")
    intuition("رأيت 6 عملاء يسددون من 8. أي احتمال سداد $p$ يجعل هذه المشاهدة أقل مفاجأة؟ جرّب $p = 0.3$: مفاجئ جدًا. $p = 0.75$: طبيعي. **تعظيم الاحتمال الأرجح** = اختيار $p$ الذي يجعل ما رأيته الأقل مفاجأة.")
    equation(r"\mathcal{L}(p) = \prod_{i=1}^{n} p^{y_i}(1-p)^{1-y_i}, \qquad \ell(p) = \log\mathcal{L}(p) = \sum_{i=1}^{n}\big[y_i\log p + (1-y_i)\log(1-p)\big]",
             [(r"\prod", "حاصل ضرب: المشاهدات مستقلة فاحتمالها المشترك حاصل ضرب احتمالاتها."), ("p^{y_i}(1-p)^{1-y_i}", "حيلة: إن كان $y_i = 1$ يبقى $p$، وإن كان 0 يبقى $1-p$."), (r"\ell", "اللوغاريتم يحوّل الضرب إلى جمع (أسهل اشتقاقًا ولا يتلاشى عدديًا).")],
             meaning_ar="لوغاريتم الاحتمال الأرجح مجموع على المشاهدات؛ تعظيمه يعني اختيار $p$ يفسر البيانات.",
             example_ar="6 من 8: $\\ell(p) = 6\\log p + 2\\log(1-p)$، أقصاه عند $p = 6/8 = 0.75$.",
             dl_link_ar="اجعل $p$ يعتمد على $x$ عبر الشبكة: $p = \\sigma(f(x))$. سالب $\\ell$ مقسومًا على $n$ هو **الإنتروبيا المتقاطعة الثنائية** حرفيًا. لم نخترع الخسارة؛ اشتققناها.", title_ar="الاحتمال الأرجح لبرنولي")
    worked_steps([("المشاهدات", r"y = (1,1,0,1,0,1,1,1),\ n=8,\ k=6"), ("اللوغاريتم", r"\ell(p) = 6\log p + 2\log(1-p)"),
                  ("اشتق وساوِ بالصفر", r"\frac{6}{p} - \frac{2}{1-p} = 0 \Rightarrow p = 0.75"), ("الخسارة عند الحل", r"\text{BCE} = -\ell(0.75)/8 \approx 0.562")])
    h3("منحنى الاحتمال الأرجح", "The likelihood curve")
    ps = np.linspace(0.01, 0.99, 200); k, n = 6, 8
    ll = k * np.log(ps) + (n - k) * np.log(1 - ps)
    fig = go.Figure(go.Scatter(x=ps, y=-ll / n, line=dict(color="#C8473A", width=3), name="BCE(p) = −ℓ(p)/n"))
    fig.add_vline(x=0.75, line=dict(color="#1F7A78", dash="dot"), annotation_text="p̂ = 0.75")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="p", yaxis_title="loss", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9")
    st.plotly_chart(fig, width="stretch", key="lik_fig")
    code_lab(CodeLab(
        key="prob_lik", title_ar="احتمال أرجح ← خسارة ثنائية، وطبيعي ← MSE", code=CODE,
        before=Before(goal_ar="حساب الاحتمال الأرجح وسالب لوغاريتمه لعدة p، التحقق من الحل التحليلي، ثم ربط BCE وMSE بالاحتمال الأرجح.", stage_ar="احتمال ← دوال الخسارة.",
                      inputs_ar="8 مشاهدات ثنائية و3 قيم انحدار.", expected_ar="أصغر −logL عند p = 0.75، BCE = nll/n، وNLL الطبيعي = SSE/(2σ²) + ثابت."),
        explain=[("6-10", "الصيغتان. لاحظ أن $\\mathcal{L}$ أرقام صغيرة جدًا (حاصل ضرب احتمالات) — لهذا نعمل باللوغاريتم."),
                 ("12-14", "جدول: −logL أصغر عند 0.75 كما يقول التحليل."),
                 ("17-19", "BCE بصيغتها المألوفة = نفس الرقم. الخسارة «المعروفة» هي الاحتمال الأرجح مقلوبًا ومقسومًا."),
                 ("22-24", "للانحدار: افتراض $y = f(x) + \\epsilon$ مع $\\epsilon \\sim N(0, \\sigma^2)$ يجعل سالب اللوغاريتم = مجموع مربعات الخطأ / 2σ² + ثابت. تقليله = تقليل MSE.")],
        run=run_printed(CODE),
        after_ar="- MSE تفترض ضمنيًا ضوضاء طبيعية متساوية التباين؛ MAE تقابل ضوضاء لابلاس. اختيار الخسارة = افتراض عن الضوضاء.\n- من هنا: Sigmoid+BCE، Softmax+CCE، خطي+MSE ثلاث حالات من مبدأ واحد.",
    ))
    why("لماذا يهم هذا الباحث؟ لأنه يجيب عن «أي خسارة أختار؟» بمبدأ لا بحفظ: صِف توزيع الهدف بشرط المدخل، خذ سالب لوغاريتم احتماله الأرجح، فهذه خسارتك.")
    common_mistake("تعظيم الاحتمال الأرجح على بيانات التدريب يمكن أن يعطي $p = 1$ أو $0$ لمشاهدة نادرة ⇒ $\\log 0$ وخسارة لانهائية على بيانات جديدة. التنظيم والتنعيم `label smoothing` يمنعان الثقة المطلقة.")
    quiz("prob.lik", [
        Q("الاحتمال الأرجح دالة في…", ["البيانات", "المعلمات", "الخسارة"], 1, "البيانات معلومة."),
        Q("BCE هي…", ["اختراع تجريبي", "سالب لوغاريتم احتمال برنولي الأرجح مقسومًا على n", "مربع الخطأ للاحتمالات"], 1, "اشتقاق لا اختراع."),
        Q("MSE تقابل افتراض ضوضاء…", ["برنولي", "طبيعية", "منتظمة"], 1, "غاوس."),
        Q("3 نجاحات من 10: $\\hat{p}$ بالاحتمال الأرجح…", ["0.3", "0.5", "0.7"], 0, "k/n.", kind="equation"),
    ])
    takeaway("Likelihood: أي معلمات تجعل المشاهد أقل مفاجأة. −log L / n = الخسارة. برنولي→BCE، فئوي→CCE، طبيعي→MSE.")
    lesson_footer(LESSON, ["الاحتمال ↔ الاحتمال الأرجح: نفس التعبير، سؤال مختلف.", "اللوغاريتم يحول الضرب إلى جمع.", "اختيار الخسارة = افتراض عن توزيع الهدف."])
