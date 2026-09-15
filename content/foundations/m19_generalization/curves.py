import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, intuition, takeaway
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto

LESSON = Lesson(
    id="foundations.generalization.curves",
    title_ar="منحنيات التدريب مقابل التحقق: ثماني حالات",
    title_en="Training vs Validation Curves: Eight Cases",
    module="foundations.generalization",
    order=1,
    prerequisites=["foundations.training_loop.validation_in_loop", "foundations.optim.learning_rate"],
    objectives_ar=["التعرف البصري على الحالات الثماني القياسية.", "استنتاج الإجراء التالي من شكل المنحنيين.", "تمييز الضوضاء الطبيعية عن الإشارة."],
    terms=["loss", "epoch"],
    labs=["labs.curves_diagnostic_lab"],
    difficulty="intermediate",
    summary_ar="اقرأ المنحنيين معًا: المستوى، الفجوة، الاتجاه، النعومة. ثماني حالات تغطي 95% مما ستراه.",
)

CASES = {
    "صحي": (lambda t: 1.0 * np.exp(-t / 8) + 0.15, lambda t: 1.0 * np.exp(-t / 8) + 0.2, "كلاهما يهبط ويستقر، الفجوة صغيرة وثابتة."),
    "فرط تخصيص": (lambda t: 1.0 * np.exp(-t / 5) + 0.02, lambda t: 1.0 * np.exp(-t / 6) + 0.2 + 0.012 * np.maximum(t - 12, 0), "التدريب يهبط باستمرار؛ التحقق يهبط ثم يرتفع: الفجوة تتسع."),
    "قصور تعلم": (lambda t: 1.0 * np.exp(-t / 30) + 0.55, lambda t: 1.0 * np.exp(-t / 30) + 0.58, "كلاهما مرتفع ومتقارب؛ القدرة أو التدريب غير كافيين."),
    "معدل تعلم صغير جدًا": (lambda t: 1.0 - 0.012 * t, lambda t: 1.02 - 0.012 * t, "هبوط خطي بطيء جدًا بلا انحناء."),
    "معدل تعلم كبير": (lambda t: 0.5 + 0.25 * np.sin(t * 1.3) + 0.1 * np.exp(-t / 40), lambda t: 0.55 + 0.25 * np.sin(t * 1.3 + 0.3), "تذبذب حاد لا يستقر."),
    "تباعد": (lambda t: 0.7 * np.exp(t / 9), lambda t: 0.75 * np.exp(t / 9), "الخسارة تكبر أسّيًا ثم NaN."),
    "تدريب ضجيج": (lambda t: 0.6 * np.exp(-t / 8) + 0.2 + 0.08 * np.sin(t * 2.1), lambda t: 0.6 * np.exp(-t / 8) + 0.28 + 0.12 * np.sin(t * 1.7 + 1), "اتجاه هابط لكن بتذبذب كبير: دفعات صغيرة أو بيانات قليلة."),
    "هضبة": (lambda t: np.where(t < 15, 1.0 - 0.02 * t, 0.7 - 0.0005 * (t - 15)), lambda t: np.where(t < 15, 1.02 - 0.02 * t, 0.72 - 0.0005 * (t - 15)), "هبوط ثم توقف طويل على مستوى مرتفع."),
}


def render() -> None:
    lesson_header(LESSON)
    h2("اقرأ الاثنين معًا", "Read both together")
    definition("منحنى الخسارة يرسم `loss` و`val_loss` مقابل الحقب. أربع خصائص تُقرأ: **المستوى** (هل الخسارة معقولة مقارنة بالمرجع؟)، **الفجوة** (تدريب مقابل تحقق)، **الاتجاه** (هبوط/ثبات/صعود)، **النعومة** (تذبذب).")
    case = st.selectbox("الحالة", list(CASES), key="curves_case")
    f_tr, f_va, desc = CASES[case]
    t = np.arange(0, 40)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t + 1, y=f_tr(t), name="loss (train)", line=dict(color="#2F6FB5", width=3)))
    fig.add_trace(go.Scatter(x=t + 1, y=f_va(t), name="val_loss", line=dict(color="#C8473A", width=3)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title=case, xaxis_title="epoch", yaxis=dict(range=[0, 1.4]), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="curves_fig")
    st.markdown(f"**ما تراه:** {desc}")
    compare_table(["الحالة", "المستوى", "الفجوة", "الاتجاه", "النعومة", "الإجراء"],
                  [("صحي", "منخفض", "صغيرة ثابتة", "هبوط ثم استقرار", "ناعم", "لا شيء؛ ربما مزيد من الحقب قليلًا"),
                   ("فرط تخصيص", "تدريب منخفض جدًا", "تتسع", "التحقق يرتفع", "ناعم", "إيقاف مبكر، تنظيم، بيانات، نموذج أصغر"),
                   ("قصور تعلم", "مرتفع", "صغيرة", "يستقر مبكرًا", "ناعم", "قدرة أكبر، حقب أكثر، معدل تعلم أكبر، خصائص أفضل"),
                   ("η صغير جدًا", "مرتفع", "صغيرة", "هبوط خطي بطيء", "ناعم", "ارفع η ×3–10"),
                   ("η كبير", "متوسط", "—", "لا يستقر", "مسنّن", "اخفض η، جدولة"),
                   ("تباعد", "ينفجر", "—", "صعود أسّي", "—", "اخفض η كثيرًا، قصّ، تحجيم، تهيئة"),
                   ("ضجيج", "هبوط", "متغيرة", "هابط بتذبذب", "خشن", "دفعة أكبر، بيانات أكثر، متوسط عدة بذور"),
                   ("هضبة", "مرتفع", "صغيرة", "توقف طويل", "ناعم", "ReduceLROnPlateau، زخم، تهيئة، تنشيط")],
                  ["rtl", "rtl", "rtl", "rtl", "rtl", "rtl"])
    intuition("قاعدة سريعة: **الفجوة** تقول فرط تخصيص (كبيرة) أو قصور (صغيرة ومرتفعة)؛ **الشكل** يقول معدل التعلم (خطي/مسنّن/منفجر)؛ **المستوى** مقارنةً بالمرجع (ln K، تباين الهدف) يقول إن كان هناك خلل أصلًا.")
    st.button("افتح معمل تشخيص المنحنيات (تحدٍّ بلا تسميات)", icon=":material/science:", type="primary", on_click=goto, args=("labs.curves_diagnostic_lab",), key="curves_lab")
    common_mistake("الحكم على «فرط تخصيص» من ارتفاع طفيف في val_loss لحقبة واحدة: ضوضاء التحقق طبيعية. ابحث عن اتجاه صاعد مستمر لعدة حقب.")
    quiz("gen.curves", [
        Q("loss يهبط إلى 0.02 وval_loss يرتفع بعد الحقبة 12…", ["قصور تعلم", "فرط تخصيص", "η كبير"], 1, "الفجوة تتسع."),
        Q("كلاهما عند 0.6 ومستقران منذ الحقبة 5…", ["فرط تخصيص", "قصور تعلم", "صحي"], 1, "مرتفع ومتقارب."),
        Q("هبوط خطي بطيء لعشرات الحقب…", ["η صغير", "η كبير", "تباعد"], 0, "لا انحناء."),
        Q("ارتفاع val_loss لحقبة واحدة ثم عودته…", ["فرط تخصيص", "ضوضاء طبيعية", "تسريب"], 1, "اتجاه لا نقطة."),
    ])
    takeaway("المستوى، الفجوة، الاتجاه، النعومة. ثماني حالات وإجراء لكل واحدة. اتجاه لا نقطة.")
    lesson_footer(LESSON, ["أربع خصائص لقراءة المنحنى.", "جدول الحالات الثماني.", "المعمل يختبرك بلا تسميات."])
