import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import definition, intuition, takeaway
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.activations.why_nonlinearity",
    title_ar="خريطة دوال التنشيط: لماذا، وأين، وأيها",
    title_en="Activation Map: Why, Where, and Which",
    module="foundations.activations",
    order=1,
    prerequisites=["foundations.neuron.activation_intro"],
    objectives_ar=["رؤية كل دوال التنشيط الرئيسية على رسم واحد.", "التمييز بين دور التنشيط في المخفية (لاخطية) وفي الإخراج (شكل التنبؤ).", "جدول القرار الذي تعود إليه دائمًا."],
    terms=["softmax", "function"],
    labs=["labs.activation_lab"],
    difficulty="beginner",
    summary_ar="المخفية: ReLU افتراضيًا. الإخراج: خطي للانحدار، Sigmoid للثنائي، Softmax للمتعدد. جدول واحد.",
)

ACTS = {
    "sigmoid": lambda z: 1 / (1 + np.exp(-z)), "tanh": np.tanh, "relu": lambda z: np.maximum(0, z),
    "leaky_relu": lambda z: np.where(z > 0, z, 0.1 * z), "elu": lambda z: np.where(z > 0, z, np.exp(np.minimum(z, 0)) - 1),
    "swish": lambda z: z / (1 + np.exp(-z)), "gelu": lambda z: 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3))),
}
COLORS = {"sigmoid": "#C8473A", "tanh": "#C77A1A", "relu": "#1F7A78", "leaky_relu": "#2E8B57", "elu": "#7C5CBF", "swish": "#2F6FB5", "gelu": "#B8860B"}


def render() -> None:
    lesson_header(LESSON)
    h2("كلها على رسم واحد", "All on one chart")
    chosen = st.multiselect("اختر الدوال", list(ACTS), default=["sigmoid", "tanh", "relu"], key="actmap_pick")
    z = np.linspace(-5, 5, 300)
    fig = go.Figure()
    for name in chosen:
        fig.add_trace(go.Scatter(x=z, y=ACTS[name](z), name=name, line=dict(color=COLORS[name], width=3)))
    fig.add_hline(y=0, line=dict(color="#B9B2A6", width=1)); fig.add_vline(x=0, line=dict(color="#B9B2A6", width=1))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="z", yaxis_title="f(z)", yaxis=dict(range=[-2, 5]), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="actmap_fig")
    definition("دور التنشيط في **الطبقة المخفية**: إدخال اللاخطية (أي دالة غير خطية تعمل، لكن بعضها يتدرب أفضل). دوره في **طبقة الإخراج**: إعطاء التنبؤ **الشكل والمدى** الذي تحتاجه المهمة والخسارة.")
    intuition("سؤالان مختلفان: «كيف أُدخل انحناءً؟» (المخفية: ReLU وعائلته) و«ماذا يجب أن يكون شكل المخرج؟» (الإخراج: احتمال؟ عدد حر؟ توزيع؟).")
    compare_table(["الدالة", "المدى", "المشتقة القصوى", "المخفية؟", "الإخراج؟", "ملاحظة"],
                  [("Sigmoid", "(0, 1)", "0.25", "نادرًا (تشبع)", "ثنائي / متعدد التسميات", "احتمال"),
                   ("tanh", "(−1, 1)", "1", "أحيانًا (RNN)", "نادرًا", "مركزها صفر"),
                   ("ReLU", "[0, ∞)", "1", "**الافتراضي**", "انحدار لقيم موجبة فقط", "سريع؛ خطر Dead ReLU"),
                   ("Leaky ReLU / ELU / SELU", "(−∞, ∞) تقريبًا", "1", "بديل لـ ReLU", "لا", "تدرج غير صفري للسالب"),
                   ("Swish / GELU", "(−0.28, ∞)", "~1.1", "الشبكات الحديثة الكبيرة", "لا", "ناعمة"),
                   ("Softmax", "احتمالات مجموعها 1", "—", "لا", "متعدد الفئات", "على متجه لا عنصر"),
                   ("خطي (بلا)", "(−∞, ∞)", "1", "لا (ينهار)", "انحدار", "عدد حر")],
                  ["ltr", "code", "num", "rtl", "rtl", "rtl"])
    quiz("act.map", [
        Q("تنشيط المخفية الافتراضي اليوم…", ["Sigmoid", "ReLU", "Softmax"], 1, "سريع وبلا إشباع للموجب."),
        Q("مخرج انحدار قد يكون سالبًا: تنشيط الإخراج…", ["ReLU", "خطي", "Sigmoid"], 1, "عدد حر."),
        Q("Softmax تُطبَّق على…", ["عنصر واحد", "متجه كامل (كل الفئات معًا)", "الطبقات المخفية"], 1, "القسمة على المجموع."),
    ])
    takeaway("المخفية: ReLU (أو بدائله). الإخراج: خطي/Sigmoid/Softmax حسب المهمة. المدى والمشتقة يفسران كل شيء.")
    lesson_footer(LESSON, ["سؤالان: لاخطية المخفية، شكل الإخراج.", "الجدول المرجعي.", "المشتقة القصوى تنبئ بالتلاشي."])
