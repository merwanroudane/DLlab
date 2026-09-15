import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, practical_note, takeaway
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson

LESSON = Lesson(
    id="foundations.optim.schedules_convergence",
    title_ar="جداول معدل التعلم والهضبة والتقارب",
    title_en="Learning-rate Schedules, Plateaus & Convergence",
    module="foundations.optim",
    order=7,
    prerequisites=["foundations.optim.optimizer_comparison"],
    objectives_ar=["الجداول الشائعة: خطوة، أسّي، جيب تمام، تقليل عند الهضبة، warm-up.", "معنى التقارب عمليًا ومتى نتوقف.", "قراءة الهضبة وتمييزها عن التقارب."],
    terms=["learning_rate", "epoch"],
    difficulty="intermediate",
    summary_ar="ابدأ كبيرًا للسرعة وانتهِ صغيرًا للدقة. ReduceLROnPlateau يخفّض عند التوقف. التقارب = لا تحسن ذي معنى على التحقق.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("لماذا نغيّر معدل التعلم أثناء التدريب؟", "Why schedule?")
    definition("**الجدول** `Schedule` دالة تعطي $\\eta_t$ بدلالة الخطوة/الحقبة. الحدس: في البداية نحن بعيدون فنريد خطوات كبيرة؛ في النهاية قرب القاع نريد خطوات صغيرة لتفادي التذبذب (رأيت هذا في درس معدل التعلم). **Warm-up**: البدء بمعدل صغير يزداد لبضع حقب لتفادي انفجار مبكر مع تهيئة عشوائية.")
    T = 100; t = np.arange(T)
    step = np.where(t < 40, 0.1, np.where(t < 70, 0.01, 0.001)); expo = 0.1 * 0.96 ** t; cos = 0.001 + 0.5 * (0.1 - 0.001) * (1 + np.cos(np.pi * t / T)); warm = np.minimum(t / 10, 1) * cos
    fig = go.Figure()
    for name, ys, c in [("step", step, "#2F6FB5"), ("exponential", expo, "#C77A1A"), ("cosine", cos, "#1F7A78"), ("warm-up + cosine", warm, "#7C5CBF")]:
        fig.add_trace(go.Scatter(x=t, y=ys, name=name, line=dict(color=c, width=2.5)))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", yaxis=dict(title="η", type="log"), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="sched_fig")
    compare_table(["الجدول", "English", "القاعدة", "Keras", "متى"],
                  [("خطوي", "Step decay", "اقسم على 10 كل k حقبة", "LearningRateScheduler", "CNN كلاسيكية"), ("أسّي", "Exponential", "η₀·γ^t", "ExponentialDecay", "بسيط"),
                   ("جيب تمام", "Cosine annealing", "من η₀ إلى ~0 على منحنى جيب تمام", "CosineDecay", "شائع حديثًا"), ("عند الهضبة", "ReduceLROnPlateau", "اقسم على factor إذا لم يتحسن val_loss لـ patience حقب", "ReduceLROnPlateau", "لا تعرف الطول مسبقًا"),
                   ("إحماء", "Warm-up", "زيادة خطية لبضع حقب ثم جدول آخر", "مخصص / CosineDecay(warmup)", "Transformers، Adam بمعدل كبير")],
                  ["rtl", "ltr", "rtl", "code", "rtl"])
    h2("الهضبة والتقارب", "Plateau & convergence")
    definition("**الهضبة** `Plateau`: توقف تحسّن الخسارة مؤقتًا (سرج، منطقة مستوية، أو η كبير يقفز حول القاع). **التقارب** `Convergence`: لا تحسّن ذي معنى على **التحقق** لعدة حقب رغم تخفيض η — نتوقف (الإيقاف المبكر، الوحدة 20).")
    compare_table(["ما تراه", "هضبة أم تقارب؟", "الإجراء"],
                  [("خسارة التدريب مستوية، التحقق مستوٍ، η ما زال كبيرًا", "هضبة محتملة", "خفّض η (ReduceLROnPlateau)"), ("مستويتان بعد تخفيض η مرتين", "تقارب", "توقف؛ الأفضل ما تحقق"),
                   ("التدريب يهبط والتحقق يرتفع", "لا هذا ولا ذاك: فرط تخصيص", "إيقاف مبكر / تنظيم"), ("مستوية منذ البداية", "ليست هضبة: خلل", "افحص η، التحجيم، التنشيط")],
                  ["rtl", "rtl", "rtl"])
    practical_note("وصفة عملية للمقرر: `Adam(1e-3)` + `ReduceLROnPlateau(factor=0.5, patience=3)` + `EarlyStopping(patience=8, restore_best_weights=True)`. تغطي 90% من الحالات الجدولية الصغيرة.")
    common_mistake("خفض η حتى 1e-7 «لأن الخسارة ما زالت تتحرك قليلًا»: التحسن أصبح أصغر من ضوضاء التحقق. التقارب يُحكم بالتحقق وبفرق ذي معنى، لا بالرقم الثالث بعد الفاصلة.")
    quiz("optim.sched", [
        Q("لماذا نخفّض η في النهاية؟", ["لتسريع التدريب", "لتفادي التذبذب حول القاع", "لتقليل الذاكرة"], 1, "خطوات صغيرة قرب الحل."),
        Q("ReduceLROnPlateau يراقب…", ["خسارة التدريب", "مقياس التحقق (val_loss)", "معيار التدرج"], 1, "التحقق."),
        Q("Warm-up مفيد لـ…", ["تفادي انفجار مبكر مع أوزان عشوائية", "تسريع النهاية", "تقليل الحقب"], 0, "البداية الحساسة."),
    ])
    takeaway("كبير أولًا، صغير أخيرًا؛ خفّض عند الهضبة؛ احكم على التقارب من التحقق. Adam + ReduceLROnPlateau + EarlyStopping وصفة كافية للمقرر.")
    lesson_footer(LESSON, ["خمسة جداول.", "هضبة ≠ تقارب ≠ فرط تخصيص.", "الوصفة العملية."])
