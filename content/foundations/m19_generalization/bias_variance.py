import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, research_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.generalization.bias_variance",
    title_ar="الانحياز والتباين",
    title_en="Bias & Variance",
    module="foundations.generalization",
    order=3,
    prerequisites=["foundations.generalization.under_overfitting", "foundations.prob.descriptive"],
    objectives_ar=["تفكيك خطأ التعميم إلى انحياز² + تباين + ضوضاء.", "ربط الانحياز بالقصور والتباين بالفرط، ورؤية المقايضة.", "تمييز هذا «الانحياز» عن انحياز الخلية b."],
    terms=["variance", "noise", "bias"],
    difficulty="intermediate",
    summary_ar="خطأ التعميم = انحياز² (نموذج بسيط جدًا) + تباين (نموذج حساس للعينة) + ضوضاء (لا تُقلَّل). التعقيد يخفض الأول ويرفع الثاني.",
)


def render() -> None:
    lesson_header(LESSON)
    definition("**الانحياز** (هنا بمعناه الإحصائي، لا انحياز الخلية $b$): الخطأ المنهجي لأن عائلة النماذج بسيطة جدًا لتمثيل الحقيقة (خط لمنحنى). **التباين**: كم يتغير النموذج المتعلَّم إذا غيّرنا عينة التدريب — نموذج معقد يتبع الضوضاء يتغير كثيرًا.")
    equation(r"\mathbb{E}\big[(\hat f(x) - y)^2\big] = \underbrace{\big(\mathbb{E}[\hat f(x)] - f(x)\big)^2}_{\text{انحياز}^2} + \underbrace{\mathbb{E}\big[(\hat f(x) - \mathbb{E}[\hat f(x)])^2\big]}_{\text{تباين}} + \underbrace{\sigma^2}_{\text{ضوضاء}}",
             [(r"\hat f", "النموذج المتعلَّم من عينة عشوائية؛ التوقع على العينات."), ("f", "الدالة الحقيقية."), (r"\sigma^2", "تباين الضوضاء: حد أدنى لا ينزل عنه أي نموذج.")],
             meaning_ar="ثلاثة مصادر للخطأ: خطأ منهجي، حساسية للعينة، وضوضاء لا مفر منها.",
             example_ar="خط مستقيم لبيانات تربيعية: انحياز عالٍ. كثيرة حدود من الدرجة 15 لعشرين نقطة: تباين عالٍ.",
             dl_link_ar="الشبكة الصغيرة/المنظَّمة بشدة: انحياز. الشبكة الكبيرة بلا تنظيم وبيانات قليلة: تباين. الإيقاف المبكر والتنظيم يقللان التباين بثمن انحياز صغير.", title_ar="التفكيك")
    h2("التجربة: نفس النموذج، عينات مختلفة", "Same model, different samples")
    deg = st.slider("درجة كثيرة الحدود (التعقيد)", 1, 12, 1, key="bv_deg")
    rng = np.random.default_rng(0)
    xs = np.linspace(-1, 1, 200); f = np.sin(2.5 * xs)
    fig = go.Figure(go.Scatter(x=xs, y=f, name="الحقيقة f(x)", line=dict(color="#2B2A28", width=3)))
    preds = []
    for s in range(6):
        r = np.random.default_rng(s + 1); x = r.uniform(-1, 1, 20); y = np.sin(2.5 * x) + r.normal(0, 0.3, 20)
        c = np.polyfit(x, y, deg); p = np.polyval(c, xs); preds.append(p)
        fig.add_trace(go.Scatter(x=xs, y=np.clip(p, -3, 3), mode="lines", line=dict(color="#C8473A", width=1), opacity=0.5, showlegend=(s == 0), name="نماذج من 6 عينات"))
    P = np.array(preds); mean_p = P.mean(0)
    fig.add_trace(go.Scatter(x=xs, y=mean_p, name="متوسط النماذج", line=dict(color="#1F7A78", width=3, dash="dot")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[-2.5, 2.5]), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="bv_fig")
    bias2 = float(np.mean((mean_p - f) ** 2)); var = float(np.mean(P.var(0)))
    st.code(f"degree {deg}:  bias² ≈ {bias2:.3f}   variance ≈ {var:.3f}   (noise σ² = 0.09)   total ≈ {bias2 + var + 0.09:.3f}", language="text")
    intuition("درجة 1: كل الخطوط الحمراء متشابهة (تباين منخفض) لكنها كلها بعيدة عن الحقيقة (انحياز عالٍ). درجة 12: كل نموذج يلتف حول ضوضاء عينته (تباين عالٍ). الوسط (3–5) يوازن.")
    research_note("في الشبكات الكبيرة جدًا لوحظت ظاهرة «النزول المزدوج»: بعد ذروة التباين، زيادة القدرة أكثر قد تخفض الخطأ ثانية (بتنظيم ضمني). لا يغيّر هذا المبدأ العملي للمقرر: ابدأ صغيرًا وراقب الفجوة.")
    common_mistake("«الانحياز» في هذه الوحدة (خطأ منهجي للنموذج) شيء، و«الانحياز b» في الخلية شيء آخر تمامًا. السياق يحدد.")
    quiz("gen.bv", [
        Q("خط مستقيم لبيانات على شكل جيب…", ["تباين عالٍ", "انحياز عالٍ", "ضوضاء"], 1, "بسيط جدًا."),
        Q("كثيرة حدود درجة 12 على 20 نقطة…", ["انحياز عالٍ", "تباين عالٍ", "مثالي"], 1, "تتبع الضوضاء."),
        Q("أي حد لا يمكن تقليله بأي نموذج؟", ["الانحياز", "التباين", "الضوضاء"], 2, "σ²."),
        Q("التنظيم عادةً…", ["يخفض التباين ويرفع الانحياز قليلًا", "يخفض الاثنين", "يرفع الاثنين"], 0, "المقايضة."),
    ])
    takeaway("خطأ = انحياز² + تباين + ضوضاء. التعقيد يقايض الأول بالثاني؛ التنظيم والبيانات يخفضان التباين؛ الضوضاء أرضية.")
    lesson_footer(LESSON, ["التفكيك الثلاثي.", "التجربة بست عينات.", "انحياز إحصائي ≠ انحياز الخلية."])
