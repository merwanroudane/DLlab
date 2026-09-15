import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import definition, research_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.math_explainer import equation
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.activations.swish_gelu",
    title_ar="تعمّق: Swish وGELU",
    title_en="Deep Dive: Swish & GELU",
    module="foundations.activations",
    order=4,
    prerequisites=["foundations.activations.relu_family"],
    objectives_ar=["صيغتا Swish وGELU وعلاقتهما بـ ReLU وSigmoid.", "معرفة أين تُستخدمان (Transformers، شبكات كبيرة) ولماذا لا تغيّر شيئًا في MLP صغيرة."],
    terms=["exponential"],
    difficulty="advanced",
    summary_ar="Swish = z·σ(z) وGELU ≈ z·Φ(z): نسختان ناعمتان من ReLU تسمحان بقيم سالبة صغيرة؛ شائعتان في النماذج الكبيرة.",
)


def render() -> None:
    lesson_header(LESSON)
    research_note("تعمّق اختياري (ملاحظة حديثة، spec §54). لا يغيّر المسار الأساسي: ReLU يبقى الافتراضي في المقرر.")
    equation(r"\text{Swish}(z) = z\,\sigma(\beta z), \qquad \text{GELU}(z) = z\,\Phi(z) \approx 0.5\,z\left[1 + \tanh\!\left(\sqrt{2/\pi}\,(z + 0.044715 z^3)\right)\right]",
             [(r"z\,\sigma(\beta z)", "المدخل مضروبًا في «بوابة» Sigmoid؛ $\\beta = 1$ هو SiLU."), (r"\Phi(z)", "دالة التوزيع التراكمي للتوزيع الطبيعي: احتمال أن يكون متغير طبيعي أقل من $z$."), ("0.044715", "ثابت التقريب بـ tanh المستخدم في الأطر لسرعته.")],
             meaning_ar="كلتاهما ≈ ReLU للقيم الكبيرة، لكنهما ناعمتان حول الصفر وتسمحان بقيم سالبة صغيرة (حد أدنى ≈ −0.28 لـ Swish و≈ −0.17 لـ GELU).",
             example_ar="$\\text{Swish}(1) = 0.73$، $\\text{Swish}(-1) = -0.27$، $\\text{GELU}(-1) = -0.16$.",
             dl_link_ar="GELU تنشيط BERT/GPT وأغلب Transformers؛ Swish في EfficientNet. في MLP صغيرة للجداول الفرق عن ReLU ضئيل.", title_ar="Swish وGELU")
    z = np.linspace(-4, 4, 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z, y=np.maximum(0, z), name="ReLU", line=dict(color="#1F7A78", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=z, y=z / (1 + np.exp(-z)), name="Swish", line=dict(color="#2F6FB5", width=3)))
    fig.add_trace(go.Scatter(x=z, y=0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3))), name="GELU", line=dict(color="#B8860B", width=3)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[-1, 4]), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="swish_fig")
    definition("الفكرة المشتركة: **بوابة ناعمة** بدل القطع الحاد. المشتقة مستمرة، والقيم السالبة الصغيرة تمرر تدرجًا غير صفري (لا موت)، والانحناء حول الصفر يُنظّم قليلًا.")
    h2("متى تهتم؟", "When does it matter?")
    st.markdown("""
- **نعم**: نماذج كبيرة (Transformers، شبكات صور عميقة) حيث أُثبت تجريبيًا تحسّن طفيف ومستقر.
- **لا**: MLP من طبقتين على بيانات جدولية — الفرق ضمن ضوضاء البذور. ابدأ بـ ReLU؛ غيّر التنشيط آخر شيء لا أول شيء.
- **الكلفة**: exp/tanh أغلى من max؛ ملحوظة فقط في الشبكات الضخمة.
""")
    quiz("act.swish", [
        Q("Swish تساوي…", ["z·σ(z)", "max(0, z)", "σ(z)"], 0, "بوابة ناعمة."),
        Q("Φ في GELU هي…", ["Sigmoid", "دالة التوزيع التراكمي الطبيعي", "tanh"], 1, "احتمال طبيعي."),
        Q("في MLP صغيرة للجداول، استبدال ReLU بـ GELU…", ["يحسّن كثيرًا", "فرق ضئيل غالبًا", "يفسد التدريب"], 1, "ضمن الضوضاء."),
    ])
    takeaway("Swish وGELU = ReLU ناعم ببوابة احتمالية؛ مفيدان في النماذج الكبيرة؛ ليسا أولوية في المقرر.")
    lesson_footer(LESSON, ["z·σ(z) وz·Φ(z).", "ناعمتان، لا موت.", "ReLU يبقى نقطة البداية."])
