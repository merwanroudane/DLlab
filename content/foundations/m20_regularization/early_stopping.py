import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, definition, practical_note, takeaway
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from labs.tinynet import TinyNet, make_moons, train

LESSON = Lesson(
    id="foundations.regularization.early_stopping",
    title_ar="الإيقاف المبكر كتنظيم",
    title_en="Early Stopping as Regularization",
    module="foundations.regularization",
    order=3,
    prerequisites=["foundations.regularization.dropout", "foundations.training_loop.validation_in_loop"],
    objectives_ar=["فهم لماذا يُعد الإيقاف المبكر تنظيمًا (يحدّ من المسافة المقطوعة في فضاء المعلمات).", "ضبط monitor وpatience وmin_delta وrestore_best_weights.", "رؤيته على تجربة حية."],
    terms=["epoch"],
    labs=["labs.overfitting_lab"],
    difficulty="beginner",
    summary_ar="أوقف التدريب عندما يتوقف تحسّن التحقق واستعد أفضل الأوزان. مجاني، فعال، ويُستخدم دائمًا.",
)


@st.cache_data(max_entries=4, show_spinner=False)
def _demo(patience: int):
    X, y = make_moons(400, noise=0.35, seed=0)
    net = TinyNet([2, 64, 64, 1], "binary", seed=0)
    h_full = train(TinyNet([2, 64, 64, 1], "binary", seed=0), X[:120], y[:120], X_val=X[120:], y_val=y[120:], epochs=250, batch_size=16, lr=0.02, optimizer="adam", seed=0)
    h_es = train(net, X[:120], y[:120], X_val=X[120:], y_val=y[120:], epochs=250, batch_size=16, lr=0.02, optimizer="adam", seed=0, early_stopping_patience=patience)
    return h_full, h_es


def render() -> None:
    lesson_header(LESSON)
    definition("**الإيقاف المبكر** `Early stopping`: راقب مقياس التحقق كل حقبة؛ إن لم يتحسن بأكثر من `min_delta` لـ `patience` حقب متتالية، أوقف التدريب وأعد الأوزان إلى أفضل حقبة. يُعد تنظيمًا لأنه يمنع المعلمات من الابتعاد كثيرًا عن التهيئة (يشبه L2 رياضيًا في الحالات الخطية).")
    patience = st.slider("patience", 2, 30, 10, key="es_patience")
    h_full, h_es = _demo(int(patience))
    ep = np.arange(1, len(h_full["loss"]) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=h_full["loss"], name="loss (بلا إيقاف)", line=dict(color="#2F6FB5", width=2)))
    fig.add_trace(go.Scatter(x=ep, y=h_full["val_loss"], name="val_loss (بلا إيقاف)", line=dict(color="#C8473A", width=2)))
    if "stopped_epoch" in h_es:
        fig.add_vline(x=h_es["stopped_epoch"], line=dict(color="#2E8B57", dash="dot"), annotation_text=f"stop @ {h_es['stopped_epoch']}")
        fig.add_vline(x=int(np.argmin(h_es["val_loss"])) + 1, line=dict(color="#1F7A78"), annotation_text="best")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key="es_fig")
    st.code(f"no early stopping: final val_loss = {h_full['val_loss'][-1]:.4f}  (best was {min(h_full['val_loss']):.4f} at epoch {int(np.argmin(h_full['val_loss'])) + 1})\n"
            f"early stopping   : stopped at {h_es.get('stopped_epoch', 'never')}, restored val_loss = {min(h_es['val_loss']):.4f}", language="text")
    compare_table(["المعلمة", "المعنى", "قيمة معقولة"],
                  [("monitor", "المقياس المراقَب", "val_loss (أنعم من الدقة)"), ("patience", "كم حقبة نصبر بلا تحسن", "5–20 حسب ضوضاء التحقق"), ("min_delta", "أصغر تحسن يُحتسب", "1e-4 إلى 1e-3"), ("restore_best_weights", "العودة إلى أفضل حقبة", "True دائمًا"), ("epochs", "الحد الأقصى", "كبير (200+) واترك القرار للإيقاف")],
                  ["code", "rtl", "rtl"])
    practical_note("صبر قصير جدًا يوقف عند هضبة مؤقتة أو ضوضاء؛ صبر طويل يهدر وقتًا لكنه آمن مع restore_best_weights. مع ReduceLROnPlateau اجعل صبر الإيقاف أطول من صبر التخفيض.")
    common_mistake("`restore_best_weights=False` (الافتراضي في بعض الإصدارات): تحصل على أوزان آخر حقبة (بعد التدهور) لا أفضلها.")
    quiz("reg.es", [
        Q("الإيقاف المبكر يراقب…", ["loss", "val_loss", "معيار التدرج"], 1, "التحقق."),
        Q("لماذا يُعد تنظيمًا؟", ["يقلل الحقب فقط", "يحدّ من ابتعاد المعلمات عن التهيئة", "يصفّر الأوزان"], 1, "مسافة محدودة."),
        Q("patience = 1 مع تحقق ضجيج…", ["مثالي", "يتوقف مبكرًا جدًا بسبب الضوضاء", "لا يتوقف"], 1, "صبر أطول."),
    ])
    takeaway("راقب val_loss، اصبر patience حقب، واستعد الأفضل. مجاني ويُستخدم دائمًا مع epochs كبير.")
    lesson_footer(LESSON, ["تجربة حية بصبر قابل للتعديل.", "المعلمات الأربع.", "restore_best_weights=True."])
