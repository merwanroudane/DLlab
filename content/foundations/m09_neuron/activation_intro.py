import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway, why
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.neuron.activation_intro",
    title_ar="التنشيط: لماذا نحتاج دالة لاخطية بعد المجموع؟",
    title_en="Activation: Why a Nonlinear Function After the Sum?",
    module="foundations.neuron",
    order=3,
    prerequisites=["foundations.neuron.weighted_sum_bias", "foundations.math.functions"],
    objectives_ar=["إثبات أن تراكب طبقات خطية يبقى خطيًا.", "فهم دالة التنشيط كـ«محوّل قرار» يعطي الشبكة قدرة على تمثيل حدود منحنية.", "التعرف المبكر على Sigmoid وReLU (تفصيلها في الوحدة 11)."],
    terms=["function"],
    labs=["labs.neuron_lab"],
    difficulty="beginner",
    summary_ar="بلا تنشيط لاخطي، ألف طبقة = طبقة واحدة. التنشيط يكسر الخطية ويمنح الشبكة قدرتها.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("الحجة", "The argument")
    equation(r"a^{(1)} = W_1 x + b_1, \quad a^{(2)} = W_2 a^{(1)} + b_2 = W_2 W_1 x + (W_2 b_1 + b_2) = W' x + b'",
             [("W_1, W_2", "مصفوفتا أوزان لطبقتين بلا تنشيط."), ("W' = W_2 W_1", "مصفوفة واحدة تكافئ الطبقتين."), ("b'", "انحياز واحد مكافئ.")],
             meaning_ar="طبقتان خطيتان = طبقة خطية واحدة. بالاستقراء: أي عدد من الطبقات الخطية = طبقة واحدة. العمق بلا تنشيط بلا فائدة.",
             example_ar="$W_1 = 2$، $W_2 = 3$ (أعداد): $a^{(2)} = 6x$ — خط مستقيم مهما كثرت الطبقات.",
             dl_link_ar="لذلك كل طبقة مخفية تتبعها دالة تنشيط لاخطية $f$: $a^{(1)} = f(W_1 x + b_1)$. الآن $W_2 f(W_1 x + b_1)$ لا يُختصر إلى خط.", title_ar="لماذا تنهار الطبقات الخطية")
    definition("**دالة التنشيط** `Activation function` دالة لاخطية تُطبَّق عنصرًا عنصرًا على المجموع الموزون: $a = f(z)$. أشهرها **Sigmoid** (تضغط إلى (0,1))، **tanh** (إلى (−1,1))، **ReLU** ($\\max(0, z)$).")
    intuition("بلا تنشيط، الخلية تقول «كلما زاد z زاد المخرج بنفس النسبة إلى الأبد». مع ReLU تقول «إن كان z سالبًا فلا شيء، وإلا مرّره». مع Sigmoid تقول «حوّله إلى احتمال». هذا «القرار» هو ما يسمح للطبقات بتركيب قرارات معقدة.")
    h2("شاهد الفرق", "See the difference")
    rng = np.random.default_rng(0)
    n = 300
    r = rng.uniform(0, 2.2, n); th = rng.uniform(0, 2 * np.pi, n)
    X = np.column_stack([r * np.cos(th), r * np.sin(th)]); y = (r > 1.3).astype(int)   # حلقة: غير قابل للفصل خطيًا
    use_act = st.toggle("تنشيط ReLU في الطبقة المخفية", value=True, key="act_toggle")
    # tiny 2-layer net trained by GD for the demo
    W1 = rng.normal(0, 1, (2, 8)); b1 = np.zeros(8); W2 = rng.normal(0, 1, (8, 1)); b2 = np.zeros(1)
    act = (lambda z: np.maximum(0, z)) if use_act else (lambda z: z)
    dact = (lambda z: (z > 0).astype(float)) if use_act else (lambda z: np.ones_like(z))
    for _ in range(600):
        z1 = X @ W1 + b1; a1 = act(z1); z2 = a1 @ W2 + b2; p = 1 / (1 + np.exp(-z2[:, 0]))
        d2 = (p - y)[:, None] / n; gW2 = a1.T @ d2; gb2 = d2.sum(0)
        d1 = (d2 @ W2.T) * dact(z1); gW1 = X.T @ d1; gb1 = d1.sum(0)
        W1 -= 0.5 * gW1; b1 -= 0.5 * gb1; W2 -= 0.5 * gW2; b2 -= 0.5 * gb2
    acc = ((p >= 0.5) == y).mean()
    gx, gy = np.meshgrid(np.linspace(-2.4, 2.4, 80), np.linspace(-2.4, 2.4, 80))
    G = np.column_stack([gx.ravel(), gy.ravel()])
    pg = 1 / (1 + np.exp(-((act(G @ W1 + b1)) @ W2 + b2)[:, 0]))
    fig = go.Figure(go.Contour(x=gx[0], y=gy[:, 0], z=pg.reshape(gx.shape), colorscale=[[0, "#E6F1FB"], [1, "#FBE6E2"]], contours=dict(start=0.5, end=0.5, size=1, coloring="fill"), showscale=False, opacity=0.6))
    fig.add_trace(go.Scatter(x=X[y == 1, 0], y=X[y == 1, 1], mode="markers", marker=dict(color="#C8473A", size=6), name="y=1 (outer)"))
    fig.add_trace(go.Scatter(x=X[y == 0, 0], y=X[y == 0, 1], mode="markers", marker=dict(color="#2F6FB5", size=6), name="y=0 (inner)"))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), title=f"{'with ReLU' if use_act else 'linear only'} — accuracy {acc:.2f}", yaxis=dict(scaleanchor="x"), paper_bgcolor="#FFFDF9", plot_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="act_fig")
    st.caption("بيانات على شكل حلقة: الداخل فئة والخارج فئة. بلا تنشيط، الشبكة (8 وحدات مخفية) لا تستطيع إلا خطًا مستقيمًا ≈ 50–60%. مع ReLU ترسم حدًا مغلقًا.")
    why("هذا هو السبب الجوهري لوجود الشبكات العصبية أصلًا: تركيب وحدات لاخطية بسيطة يقرّب أي دالة مستمرة تقريبًا (نظرية التقريب الشامل). الطبقات الخطية وحدها لا تقرّب إلا الخطوط.")
    common_mistake("وضع تنشيط لاخطي على **طبقة الإخراج** في الانحدار (مثل ReLU) فيصبح النموذج عاجزًا عن التنبؤ بقيم سالبة. طبقة الإخراج تنشيطها يحدده نوع الهدف لا الرغبة في اللاخطية.")
    quiz("neuron.act", [
        Q("3 طبقات Dense بلا تنشيط تكافئ…", ["3 طبقات", "طبقة واحدة", "لا شيء"], 1, "خطي ∘ خطي = خطي."),
        Q("دور دالة التنشيط…", ["تسريع التدريب", "إدخال اللاخطية", "تحجيم المدخلات"], 1, "كسر الخطية."),
        Q("بيانات على شكل حلقة تحتاج…", ["خلية واحدة", "طبقة مخفية بتنشيط لاخطي", "تحجيمًا فقط"], 1, "حد منحنٍ."),
    ])
    takeaway("خطي ∘ خطي = خطي. التنشيط اللاخطي بعد كل طبقة مخفية هو ما يعطي الشبكة قدرتها على الحدود المنحنية. تنشيط الإخراج يحدده الهدف.")
    lesson_footer(LESSON, ["برهان انهيار الطبقات الخطية.", "a = f(z) عنصرًا عنصرًا.", "الحلقة: مثال لا يُفصل خطيًا."])
