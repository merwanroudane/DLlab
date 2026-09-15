import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, debugging_note, definition, intuition, practical_note, research_note, takeaway
from components.comparison import compare_table
from components.diagram import diagram, svg_arrow, svg_box, svg_defs, svg_text
from components.lesson_layout import h2, h3, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from labs.fw import keras_tensorboard_run

LESSON = Lesson(
    id="foundations.frameworks.tensorflow.tensorboard_errors",
    title_ar="TensorBoard ومراقبة التدريب، وأخطاء TensorFlow الشائعة",
    title_en="TensorBoard / Training Monitoring & Common TensorFlow Errors",
    module="foundations.frameworks",
    parent="foundations.frameworks.tensorflow",
    order=18,
    prerequisites=["foundations.frameworks.tensorflow.keras_custom_loop", "foundations.generalization.curves"],
    objectives_ar=["فهم TensorBoard كأداة تصوّر/مراقبة تقرأ ملفات سجلات — وليست TensorFlow نفسه.", "قراءة كل لوحة: Scalars (خسارة/دقة/معدل التعلم)، Graphs، Histograms (تعميق).", "تشخيص أخطاء الأجهزة وdtype والأشكال في TensorFlow."],
    terms=["epoch", "learning_rate", "dtype", "shape"],
    labs=["labs.curves_diagnostic_lab"],
    difficulty="intermediate",
    summary_ar="TensorBoard يقرأ ملفات events التي يكتبها استدعاء TensorBoard (أو tf.summary) ويعرض Scalars/Graphs/Histograms. تشغيله: tensorboard --logdir logs. أخطاء TensorFlow: اقرأ آخر سطر، طابق dtype والشكل والجهاز.",
)

WRITE = '''tb = keras.callbacks.TensorBoard(log_dir="logs/run1", histogram_freq=1)   # يكتب ملفات events كل حقبة
model.fit(X, y, validation_data=(X_val, y_val), epochs=30, callbacks=[tb])
# ثم في الطرفية (أو في Colab: %load_ext tensorboard ثم %tensorboard --logdir logs):
#   tensorboard --logdir logs      ->  http://localhost:6006'''


def _flow_svg() -> str:
    s = '<svg viewBox="0 0 760 120" width="100%" style="max-width:760px">' + svg_defs()
    boxes = [("model.fit(...)", "#E3F3F0"), ("TensorBoard callback / tf.summary", "#E3F3F0"), ("logs/run1/events.out.tfevents.*", "#FFF3D6"), ("tensorboard --logdir logs", "#FFF3D6"), ("browser :6006", "#E6F1FB")]
    x = 10
    widths = [110, 190, 200, 150, 90]
    for (lbl, fill), w in zip(boxes, widths):
        s += svg_box(x, 35, w, 46, lbl, fill, font=11, bold=True)
        x += w + 12
        if lbl != "browser :6006":
            s += svg_arrow(x - 11, 58, x - 1, 58)
    s += svg_text(380, 105, "TensorFlow / Keras write files  →  TensorBoard (a separate program) reads them", size=11, color="#6B675F")
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("ما هو TensorBoard؟", "What is TensorBoard?")
    definition("**TensorBoard**: برنامج **منفصل** لتصوّر سجلات التدريب. TensorFlow/Keras (أو PyTorch عبر `torch.utils.tensorboard`) **يكتب** ملفات أحداث على القرص؛ TensorBoard **يقرؤها** ويعرضها في المتصفح. ليس جزءًا من الحساب ولا من النموذج — إن حذفته لا يتغير التدريب.")
    diagram("من fit إلى المتصفح", _flow_svg(), what_ar="خمس محطات: التدريب يكتب، TensorBoard يقرأ.", how_ar="الملف `events.out.tfevents.*` هو الوسيط. مجلد لكل تشغيل (run) يتيح مقارنة عدة تجارب في لوحة واحدة.", takeaway_ar="TensorBoard أداة مراقبة تعمل مع أي إطار يكتب بصيغتها.", title_en="Logging flow")
    st.code(WRITE, language="python")
    ep = st.slider("حقب التشغيل المسجَّل", 4, 20, 12, key="tb_ep")
    r = keras_tensorboard_run(int(ep))
    with st.expander("الملفات التي كُتبت فعليًا في هذا التشغيل", icon=":material/folder:"):
        st.code(f"log_dir = {r['log_dir']}\n" + "\n".join(r["files"]), language="text")
        st.markdown("مجلد `train/` ومجلد `validation/`: TensorBoard يرسم كل واحد كمنحنى مستقل بلون مختلف على نفس اللوحة.")
    practical_note("اللوحات أدناه **إعادة بناء تعليمية** لما يعرضه TensorBoard، مرسومة من نفس البيانات التي كُتبت في الملفات أعلاه — حتى تتعلم قراءة كل لوحة هنا قبل فتح الأداة الحقيقية.")
    hist = r["history"]; e = np.arange(1, len(hist["loss"]) + 1)
    h2("لوحة Scalars", "Scalars panel")
    c1, c2 = st.columns(2)
    with c1:
        f = go.Figure()
        f.add_trace(go.Scatter(x=e, y=hist["loss"], name="train (epoch_loss)", line=dict(color="#E8710A", width=2.5)))
        f.add_trace(go.Scatter(x=e, y=hist["val_loss"], name="validation (epoch_loss)", line=dict(color="#12B5CB", width=2.5)))
        f.update_layout(height=280, title="epoch_loss", margin=dict(l=10, r=10, t=35, b=10), xaxis_title="epoch (step)", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
        st.plotly_chart(f, width="stretch", key="tb_loss")
    with c2:
        f = go.Figure()
        f.add_trace(go.Scatter(x=e, y=hist["accuracy"], name="train", line=dict(color="#E8710A", width=2.5)))
        f.add_trace(go.Scatter(x=e, y=hist["val_accuracy"], name="validation", line=dict(color="#12B5CB", width=2.5)))
        f.update_layout(height=280, title="epoch_accuracy", margin=dict(l=10, r=10, t=35, b=10), xaxis_title="epoch (step)", yaxis=dict(range=[0, 1]), plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
        st.plotly_chart(f, width="stretch", key="tb_acc")
    f = go.Figure(go.Scatter(x=e, y=r["lr"], name="learning_rate", line=dict(color="#7C5CBF", width=2.5), mode="lines+markers"))
    f.update_layout(height=220, title="epoch_learning_rate", margin=dict(l=10, r=10, t=35, b=10), xaxis_title="epoch (step)", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9")
    st.plotly_chart(f, width="stretch", key="tb_lr")
    with st.expander("كيف أقرأ لوحة Scalars؟", icon=":material/visibility:", expanded=True):
        st.markdown("""
- **بطاقة لكل كمية**: `epoch_loss`, `epoch_accuracy`, `epoch_learning_rate` (تظهر عندما تستخدم جدولًا/استدعاء LR). المحور الأفقي **step** = الحقبة (أو الدفعة إن سجّلت `update_freq='batch'`).
- **لونان لكل بطاقة**: برتقالي `train`، أزرق `validation` — من المجلدين. الفجوة بينهما هي فجوة التعميم (وحدة 19).
- **Smoothing** (شريط في اليسار): متوسط متحرك أسي يخفي الضوضاء؛ ضعه 0 لرؤية القيم الحقيقية.
- **عدة تشغيلات** (`logs/run1`, `logs/run2`): كل واحد بلون؛ قارن η أو البنية.
- **معدل التعلم**: ينخفض هنا بـ 0.85 كل حقبة (LearningRateScheduler). هبوط مفاجئ = ReduceLROnPlateau عمل. ثابت = لا جدول.
- **ما يثير القلق**: validation يصعد بينما train ينزل (فرط تخصيص)، كلاهما ثابت مرتفع (قصور/η)، قفزات عنيفة (η كبير، دفعات صغيرة).
""")
    h2("لوحة Graphs", "Graphs panel")
    cols = st.columns(len(r["layers"]) + 1)
    with cols[0]:
        st.markdown("**Input**\n\n`(None, 2)`")
    for c, (name, typ, shp) in zip(cols[1:], r["layers"]):
        with c:
            st.markdown(f"**{name}**\n\n{typ}\n\n`{shp}`")
    with st.expander("كيف أقرأ لوحة Graphs؟", icon=":material/visibility:"):
        st.markdown("""
- تعرض **الرسم الحسابي** للنموذج: عقد (طبقات/عمليات) وأسهم (موترات) مع أشكالها.
- انقر على عقدة مزدوجًا لتوسيعها: Dense تتحول إلى MatMul + BiasAdd + Relu.
- استخدمها للتحقق من أن **الرسم يطابق ما قصدت**: طبقة ناقصة، تفرّع خاطئ، شكل غير متوقع.
- الرسم يظهر فقط عندما يكون `fit` قد تتبّع الدالة (الافتراضي)؛ في الحلقات المخصصة الفورية لا يوجد رسم ما لم تستخدم `tf.function`.
""")
    h2("لوحة Histograms (تعميق)", "Histograms panel (deep dive)")
    W = np.array(r["weights"])
    f = go.Figure()
    for i in range(0, len(W), max(1, len(W) // 6)):
        hist_vals, edges = np.histogram(W[i], bins=25, range=(-1.5, 1.5))
        f.add_trace(go.Scatter(x=(edges[:-1] + edges[1:]) / 2, y=hist_vals + i * 0.0, name=f"epoch {i + 1}", mode="lines", line=dict(width=2), fill="tozeroy", opacity=0.35))
    f.update_layout(height=300, title="dense/kernel — distribution of the 64 weights per epoch", margin=dict(l=10, r=10, t=35, b=10), xaxis_title="weight value", yaxis_title="count", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
    st.plotly_chart(f, width="stretch", key="tb_hist")
    research_note("**كيف أقرأ Histograms؟** لكل طبقة توزيع أوزانها (وتدرجاتها إن سُجّلت) عبر الحقب، مكدّسًا كشرائح. توزيع يتسع تدريجيًا = تعلّم طبيعي. توزيع ينهار إلى صفر = أوزان ميتة/تدرج متلاشٍ (وحدة 14). توزيع ينفجر = η كبير أو انفجار تدرج. `histogram_freq=1` يكتبها كل حقبة (يكبّر الملفات).")
    h2("أخطاء TensorFlow الشائعة", "Common TensorFlow errors")
    compare_table(["الرسالة (مختصرة)", "النوع", "السبب", "التشخيص/الحل"],
                  [("`cannot compute AddV2 as input #1 was expected to be a int32 tensor but is a float tensor`", "dtype", "خلط أعداد صحيحة وكسور", "اطبع `.dtype` للطرفين؛ `tf.cast` إلى float32"),
                   ("`Incompatible shapes: [32,1] vs. [32]`", "شكل", "y بشكل (n,) ومخرج (n,1) في خسارة يدوية", "`y[:, None]` أو `tf.squeeze`; طابق الرتبة"),
                   ("`Matrix size-incompatible: In[0]: [32,4], In[1]: [5,8]`", "شكل", "matmul بأبعاد داخلية مختلفة (4 ≠ 5)", "(a,b)@(b,c): تحقق من b"),
                   ("`Could not load dynamic library 'cudart64_110.dll'`", "جهاز", "لا CUDA مثبت", "**تحذير** لا خطأ: يعمل على CPU. لـ GPU: ثبّت TensorFlow المتوافق مع CUDA أو استخدم Colab"),
                   ("`ResourceExhaustedError: OOM when allocating tensor with shape[...]`", "جهاز", "ذاكرة GPU ممتلئة", "قلّل `batch_size`، صغّر النموذج، `mixed precision` (الأسبوع 13)"),
                   ("`tape.gradient(...)` يعيد `None`", "اشتقاق", "المتغير غير مراقَب أو خارج with أو قُطع التدفق", "درس GradientTape: الأسباب الثلاثة"),
                   ("`InvalidArgumentError: Graph execution error` … `Detected at node …`", "تنفيذ", "خطأ داخل الرسم المجمَّع (fit)", "اقرأ السطر بعد `Detected at node` ثم الرسالة الأصلية أسفله"),
                   ("`loss: nan`", "عددي", "انفجار، log(0)، مدخل غير محجّم", "`TerminateOnNaN`، η أصغر، تحجيم، `from_logits=True` بدل sigmoid يدوي")],
                  ["code", "rtl", "rtl", "rtl"])
    h3("تشخيص الجهاز", "Device diagnostics")
    st.code("""import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))     # [] = لا GPU مرئي
print(tf.test.is_built_with_cuda())               # هل نسخة TensorFlow مبنية بدعم CUDA؟
x = tf.random.normal((1000, 1000)); print(x.device)   # أين وُضع الموتر فعليًا
tf.debugging.set_log_device_placement(True)       # يطبع جهاز كل عملية (للتشخيص فقط)""", language="python")
    debugging_note("«GPU موجود في Colab لكن TensorFlow لا يراه»: Runtime → Change runtime type → GPU، ثم أعد تشغيل الجلسة. تحقق بـ `list_physical_devices('GPU')` **قبل** بناء النموذج.")
    intuition("أخطاء TensorFlow ثلاث عائلات: **dtype** (طابق النوعين)، **شكل** (طابق الرتبة والأبعاد الداخلية)، **جهاز** (تحذير غالبًا، أو OOM). القراءة من آخر سطر تحدد العائلة في ثوانٍ.")
    common_mistake("«TensorBoard لا يعرض شيئًا»: المسار في `--logdir` يجب أن يحوي مجلدات التشغيل (`logs/` لا `logs/run1/train/`)، والملفات تُكتب في نهاية كل حقبة — لا تفتحه قبل انتهاء الحقبة الأولى، واضغط تحديث.")
    quiz("tf.tb", [
        Q("TensorBoard…", ["جزء من حساب TensorFlow", "برنامج منفصل يقرأ ملفات سجلات", "طبقة Keras"], 1, "أداة مراقبة."),
        Q("في لوحة Scalars، اللونان البرتقالي والأزرق…", ["حقبتان", "train وvalidation", "loss وaccuracy"], 1, "من مجلدين."),
        Q("`Matrix size-incompatible: [32,4] vs [5,8]`:", ["batch_size خطأ", "البعد الداخلي 4 ≠ 5", "dtype"], 1, "قاعدة matmul."),
        Q("`Could not load dynamic library cudart…`", ["خطأ يوقف التدريب", "تحذير: يعمل على CPU", "OOM"], 1, "لا CUDA."),
    ])
    takeaway("TensorBoard يقرأ ملفات events يكتبها استدعاء TensorBoard: Scalars (loss/accuracy/lr لكل run)، Graphs (الرسم)، Histograms (توزيعات الأوزان). أخطاء TensorFlow: dtype، شكل، جهاز — اقرأ آخر سطر.")
    lesson_footer(LESSON, ["تدفق التسجيل وقراءة اللوحات الثلاث.", "جدول الأخطاء بعائلاتها.", "تشخيص الجهاز."])
