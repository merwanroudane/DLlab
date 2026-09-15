import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, practical_note, research_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from labs.tinynet import TinyNet, make_moons, train

LESSON = Lesson(
    id="foundations.generalization.complexity_data_size",
    title_ar="تعقيد النموذج وحجم البيانات: منحنيا التعلم",
    title_en="Model Complexity & Dataset Size: Learning Curves",
    module="foundations.generalization",
    order=4,
    prerequisites=["foundations.generalization.bias_variance"],
    objectives_ar=["قراءة منحنى التعلم (الخطأ مقابل حجم البيانات) لتقرير: هل أحتاج بيانات أكثر أم نموذجًا أفضل؟", "قراءة منحنى التعقيد (الخطأ مقابل القدرة) لاختيار الحجم.", "تجربة حية على شبكة صغيرة."],
    terms=["observation"],
    labs=["labs.overfitting_lab"],
    difficulty="intermediate",
    summary_ar="منحنى التعلم: فجوة كبيرة تتقلص مع البيانات = تحتاج بيانات؛ خطأ مرتفع متقارب لا يتحسن = تحتاج نموذجًا أفضل.",
)


@st.cache_data(max_entries=4, show_spinner=False)
def _learning_curve(width: int):
    X, y = make_moons(1200, noise=0.3, seed=0)
    Xva, yva = X[1000:], y[1000:]
    sizes = [20, 40, 80, 160, 320, 640, 1000]
    tr_err, va_err = [], []
    for n in sizes:
        net = TinyNet([2, width, width, 1], "binary", seed=0)
        h = train(net, X[:n], y[:n], X_val=Xva, y_val=yva, epochs=150, batch_size=32, lr=0.1, optimizer="adam", seed=0)
        tr_err.append(1 - net.metric(X[:n], y[:n])); va_err.append(1 - net.metric(Xva, yva))
    return sizes, tr_err, va_err


def render() -> None:
    lesson_header(LESSON)
    h2("منحنى التعلم", "Learning curve")
    st.markdown("**منحنى التعلم**: خطأ التدريب والتحقق مقابل **عدد ملاحظات التدريب** (بنفس النموذج). يجيب عن أهم سؤال عملي: هل تستحق البيانات الإضافية كلفتها؟")
    width = st.select_slider("عرض الشبكة (القدرة)", options=[4, 16, 64], value=16, key="lc_width")
    sizes, tr, va = _learning_curve(int(width))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sizes, y=tr, name="خطأ التدريب", line=dict(color="#2F6FB5", width=3), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=sizes, y=va, name="خطأ التحقق", line=dict(color="#C8473A", width=3), mode="lines+markers"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), xaxis=dict(title="training set size", type="log"), yaxis=dict(title="error rate", range=[0, 0.5]), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="lc_fig")
    st.markdown("""
- **فجوة كبيرة عند بيانات قليلة تتقلص مع الزيادة** (المنحنيان يتقاربان): تباين — البيانات الإضافية تفيد.
- **المنحنيان متقاربان ومرتفعان ولا يتحسنان**: انحياز — البيانات الإضافية لن تفيد؛ حسّن النموذج/الخصائص.
- **التحقق ما زال يهبط عند آخر نقطة**: لم تشبع بعد؛ المزيد يفيد.
""")
    research_note("في البحوث الاقتصادية غالبًا لا يمكن «جمع المزيد». منحنى التعلم يخبرك مبكرًا إن كنت في منطقة الانحياز (فيُجدي تحسين النموذج) أو التباين (فتُجدي البساطة والتنظيم).")
    h2("منحنى التعقيد", "Complexity curve")
    st.markdown("الخطأ مقابل **القدرة** (عدد الوحدات/الطبقات/درجة كثيرة الحدود) بحجم بيانات ثابت: خطأ التدريب يهبط دائمًا، وخطأ التحقق يهبط ثم يصعد — القاع هو التعقيد المناسب لهذا الحجم. رأيت هذا المنحنى في درس الانحياز والتباين مع كثيرة الحدود.")
    practical_note("قاعدة عملية: لكل مضاعفة في حجم البيانات يمكن (لا يجب) زيادة القدرة. ابدأ صغيرًا، ارسم منحنى التعلم مبكرًا، وقرر أين تنفق الوقت: بيانات أم نموذج.")
    common_mistake("جمع بيانات مكلفة عندما يكون المنحنيان متقاربين ومرتفعين (انحياز): المزيد من نفس الخصائص الضعيفة لا يعلّم أكثر.")
    quiz("gen.lc", [
        Q("فجوة كبيرة تتقلص مع زيادة البيانات…", ["انحياز", "تباين: البيانات تفيد", "ضوضاء"], 1, "الفجوة تباين."),
        Q("المنحنيان متقاربان عند 30% خطأ ولا يتحسنان…", ["اجمع بيانات", "حسّن النموذج/الخصائص", "أوقف التدريب مبكرًا"], 1, "انحياز."),
        Q("منحنى التعقيد: خطأ التحقق…", ["يهبط دائمًا", "يهبط ثم يصعد", "ثابت"], 1, "قاع = التعقيد المناسب."),
    ])
    takeaway("ارسم منحنى التعلم مبكرًا: الفجوة تقول بيانات، الارتفاع المتقارب يقول نموذج. منحنى التعقيد يحدد الحجم لحجم البيانات.")
    lesson_footer(LESSON, ["منحنى التعلم على شبكة حقيقية.", "ثلاث قراءات.", "بيانات أم نموذج؟ قرار مبني على الرسم."])
