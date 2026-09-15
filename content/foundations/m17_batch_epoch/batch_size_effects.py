import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, practical_note, research_note, takeaway
from components.comparison import compare_table
from components.diagnostics import Problem, problem_card
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from labs.tinynet import TinyNet, make_moons, train

LESSON = Lesson(
    id="foundations.batch_epoch.batch_size_effects",
    title_ar="أثر حجم الدفعة: الضوضاء والذاكرة ومعدل التعلم والتعميم والحالات الحدّية",
    title_en="Batch-size Effects: Noise, Memory, Learning Rate, Generalization & Edge Cases",
    module="foundations.batch_epoch",
    order=2,
    prerequisites=["foundations.batch_epoch.definitions", "foundations.optim.gd_variants"],
    objectives_ar=["رؤية أثر batch_size على ضوضاء الخسارة ووقت الحقبة بتجربة حية.", "قاعدة الربط بين حجم الدفعة ومعدل التعلم.", "الحالات الحدّية: 1، n، أكبر من n، Batch Normalization مع دفعة صغيرة، الذاكرة."],
    terms=["batch_size", "learning_rate"],
    labs=["labs.gradient_descent_lab"],
    difficulty="intermediate",
    summary_ar="دفعة صغيرة: ضوضاء وتحديثات كثيرة وذاكرة قليلة؛ كبيرة: تدرج ناعم وتوازي وذاكرة كبيرة وقد تحتاج η أكبر. 32–256 نطاق عملي.",
)


@st.cache_data(max_entries=8, show_spinner=False)
def _curves(seed: int):
    X, y = make_moons(600, seed=0)
    out = {}
    for bs in (2, 16, 64, 480):
        net = TinyNet([2, 16, 1], "binary", seed=seed)
        out[bs] = train(net, X[:480], y[:480], X_val=X[480:], y_val=y[480:], epochs=25, batch_size=bs, lr=0.2, seed=seed)["loss"]
    return out


def render() -> None:
    lesson_header(LESSON)
    h2("التجربة", "The experiment")
    curves = _curves(0)
    fig = go.Figure()
    for bs, c in zip(curves, ["#C8473A", "#C77A1A", "#1F7A78", "#2F6FB5"]):
        fig.add_trace(go.Scatter(y=curves[bs], x=np.arange(1, 26), name=f"batch_size={bs}", line=dict(color=c, width=2.5)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", yaxis_title="train loss (epoch mean)", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="bse_fig")
    st.caption("نفس الشبكة وη = 0.2 وبذرة واحدة، 25 حقبة. الدفعة 2: 240 تحديثًا/حقبة؛ الدفعة 480: تحديث واحد/حقبة.")
    compare_table(["حجم الدفعة", "ضوضاء التدرج", "تحديثات/حقبة", "زمن الحقبة", "ذاكرة", "η المناسب", "التعميم (تجريبيًا)"],
                  [("صغير (1–16)", "عالية", "كثيرة", "طويل (لا توازي)", "قليلة", "أصغر", "جيد غالبًا (الضوضاء تنظّم)"), ("متوسط (32–256)", "متوسطة", "متوسطة", "قصير على GPU", "متوسطة", "افتراضي", "الأفضل عمليًا"),
                   ("كبير (1k+)", "منخفضة", "قليلة", "أقصر لكل حقبة", "كبيرة (قد تفيض)", "أكبر (قاعدة خطية/جذرية)", "قد يضعف؛ يحتاج warm-up وحيلًا")],
                  ["rtl", "rtl", "rtl", "rtl", "rtl", "rtl", "rtl"])
    research_note("قاعدة تجريبية شائعة: عند مضاعفة حجم الدفعة k مرة، ضاعف معدل التعلم بنحو k (الخطية) أو √k — مع warm-up. صالحة تقريبيًا في نطاق، وليست قانونًا.")
    h2("الحالات الحدّية", "Edge cases")
    compare_table(["الحالة", "ماذا يحدث", "ملاحظة"],
                  [("batch_size = 1", "SGD خالص؛ خسارة لكل ملاحظة؛ Batch Normalization لا يعمل (انحراف صفر)", "بطيء جدًا على GPU"), ("batch_size = n", "Batch GD؛ تحديث واحد لكل حقبة؛ منحنى ناعم", "الذاكرة = كل البيانات"),
                   ("batch_size > n", "تُقلَّص إلى n بصمت في أغلب الأطر", "ليست خطأ لكنها التباس"), ("n % batch_size ≠ 0", "دفعة أخيرة جزئية (أو تُسقط)", "خسارتها أكثر ضوضاء؛ الخلط يوزّعها"),
                   ("دفعة صغيرة + Batch Normalization", "إحصاءات الدفعة غير موثوقة", "استخدم ≥ 16 أو Layer/Group Norm"), ("دفعة كبيرة على GPU", "OOM: نفاد الذاكرة", "قلّل الدفعة أو استخدم تراكم التدرج")],
                  ["code", "rtl", "rtl"])
    problem_card(Problem(
        key="bad_batch_size", name_ar="حجم دفعة غير مناسب", name_en="Bad batch size",
        description_ar="حجم الدفعة يؤثر في الضوضاء والذاكرة ومعدل التعلم المناسب؛ اختيار غير مناسب يظهر كبطء أو تذبذب أو OOM.",
        symptoms_ar=["منحنى خسارة مسنّن جدًا (صغير مع η كبير).", "حقب بطيئة جدًا على GPU (صغير).", "OOM عند أول خطوة (كبير).", "تعميم أضعف رغم خسارة تدريب منخفضة (كبير جدًا)."],
        sees_ar=["`ResourceExhaustedError` / `CUDA out of memory`.", "`step` time كبير مع دفعة 8.", "loss يقفز حقبة بعد حقبة."],
        possible_causes_ar=["η لم يُعدَّل مع تغيير الدفعة.", "ذاكرة GPU أصغر من التنشيطات المطلوبة.", "دفعة صغيرة جدًا مع BatchNorm."],
        root_causes_ar=["الدفعة تحدد تباين تقدير التدرج وحجم التنشيطات المحفوظة."],
        diagnosis_ar=["اختبر 3 أحجام (16، 64، 256) بمعدل تعلم مضبوط لكل منها.", "راقب الذاكرة عند أكبر حجم.", "قارن بالتحديثات لا بالحقب."],
        evidence_ar=["ضبط η مع الحجم يزيل التذبذب؛ تقليل الحجم يزيل OOM."],
        fixes_ar=["ابدأ بـ 32 أو 64.", "عدّل η مع الحجم (خطيًا/جذريًا).", "تراكم التدرج لمحاكاة دفعة كبيرة بذاكرة صغيرة.", "≥ 16 مع BatchNorm."],
        misdiagnosis_ar=["«النموذج كبير جدًا» عند OOM بينما الدفعة هي السبب."],
        related_ar=["معدل التعلم", "الذاكرة/GPU", "Batch Normalization"],
        checklist_ar=["batch_size بين 16 و256؟", "η مضبوط لهذا الحجم؟", "الذاكرة كافية؟"],
    ))
    practical_note("للمقرر: 32 للجداول الصغيرة، 64–128 للصور على Colab GPU، 32–64 للتسلسلات. غيّرها فقط عند وجود سبب (ذاكرة، سرعة، تذبذب).")
    common_mistake("زيادة الدفعة من 32 إلى 1024 «لتسريع التدريب» دون رفع η: التحديثات تقل 32 مرة والنموذج يتعلم ببطء شديد رغم أن كل حقبة أسرع.")
    quiz("be.effects", [
        Q("مضاعفة الدفعة 4 مرات مع ثبات η…", ["نفس التعلم", "تحديثات أقل 4 مرات: أبطأ تعلمًا لكل حقبة", "أسرع تعلمًا"], 1, "عدد التحديثات."),
        Q("`CUDA out of memory` — أول إجراء…", ["تقليل الطبقات", "تقليل batch_size", "تغيير المحسّن"], 1, "التنشيطات."),
        Q("Batch Normalization مع batch_size = 2…", ["ممتاز", "إحصاءات غير موثوقة", "لا فرق"], 1, "عينة صغيرة جدًا."),
    ])
    takeaway("الدفعة تضبط الضوضاء والذاكرة والتحديثات. 32–256 عمليًا؛ عدّل η معها؛ راقب الحالات الحدّية.")
    lesson_footer(LESSON, ["تجربة حية بأربعة أحجام.", "جدول المقايضات.", "مشكلة بإطار كامل."])
