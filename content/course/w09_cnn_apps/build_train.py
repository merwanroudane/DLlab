import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.callouts import common_mistake, intuition, practical_note, takeaway, why
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go as goto
from core.rtl import table
from labs.cnn import shapes_dataset
from labs.fw import cnn_shapes_run

LESSON = Lesson(
    id="course.w09.build_train",
    title_ar="معالجة الصور مسبقًا، بناء CNN، التدريب، والتقييم",
    title_en="Image Preprocessing, Building the CNN, Training & Evaluation",
    module="course.w09",
    order=2,
    prerequisites=["course.w09.overview", "course.w08.layers_shapes", "course.w03.first_network"],
    objectives_ar=["تجهيز الصور: تحجيم 0–1، بُعد القناة، الدفعة، الهدف كأعداد صحيحة.", "بناء CNN صغيرة بالأسئلة الأحد عشر، تدريبها وقراءة summary والسجل.", "التقييم: دقة الاختبار ومصفوفة التباس لثلاث فئات."],
    terms=["tensor", "cross_entropy", "batch_size"],
    labs=["labs.cnn_shape_calculator", "labs.confusion_matrix_lab"],
    difficulty="intermediate",
    summary_ar="X (n, 16, 16, 1) في [0,1]، y أعداد صحيحة 0..2. CNN: [Conv 8 → Pool → Conv 16 → Pool] → Flatten → Dense 32 → Dense 3 softmax، sparse_categorical_crossentropy، Adam. التقييم بمصفوفة التباس.",
)

CODE = '''# الكود الذي تشغّله هذه الصفحة (labs/fw.py: cnn_shapes_run)
X, y = shapes_dataset(n=600, size=16, noise=0.2)                     # X: (600, 16, 16, 1) float32 في [0, 1] ; y: (600,) int64 ∈ {0, 1, 2}
X_tr, y_tr, X_va, y_va, X_te, y_te = X[:300], y[:300], X[300:450], y[300:450], X[450:], y[450:]

model = keras.Sequential([
    layers.Input(shape=(16, 16, 1)),                                  # صورة واحدة: H, W, C
    layers.Conv2D(8, 3, padding="same", activation="relu"),           # (16, 16, 8)   80 معلمة
    layers.MaxPooling2D(2),                                           # (8, 8, 8)
    layers.Conv2D(16, 3, padding="same", activation="relu"),          # (8, 8, 16)    1,168
    layers.MaxPooling2D(2),                                           # (4, 4, 16)
    layers.Flatten(),                                                 # (256,)
    layers.Dense(32, activation="relu"),                              # 8,224
    layers.Dense(3, activation="softmax"),                            # 99  -> 3 احتمالات
])
model.compile(optimizer=keras.optimizers.Adam(2e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=8, batch_size=32)
test_loss, test_acc = model.evaluate(X_te, y_te)
pred = model.predict(X_te).argmax(axis=1)                            # (150, 3) احتمالات -> فئة'''


def render() -> None:
    lesson_header(LESSON)
    h2("1) البيانات ومعالجتها", "1) Data & preprocessing")
    X, y = shapes_dataset(8, size=16, seed=5, noise=0.2)
    cols = st.columns(8)
    for c, i in zip(cols, range(8)):
        with c:
            fig = go.Figure(go.Heatmap(z=X[i, ::-1, :, 0], colorscale="Greys", showscale=False))
            fig.update_layout(height=110, margin=dict(l=0, r=0, t=16, b=0), title=dict(text=["h-bar", "v-bar", "cross"][y[i]], font=dict(size=10), x=0.5), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), paper_bgcolor="#FFFDF9")
            st.plotly_chart(fig, width="stretch", key=f"w09_s{i}")
    compare_table(["الخطوة", "لماذا", "الكود"],
                  [("القيم إلى [0, 1]", "الشبكة تتدرب على مقاييس صغيرة (الأسس 8)؛ الصور الحقيقية 0–255", "`X = X.astype('float32') / 255`"), ("بُعد القناة", "Conv2D يتوقع (H, W, C) حتى للرمادي", "`X = X[..., None]` → (n, 16, 16, 1)"), ("الدفعة", "رتبة 4 (B, H, W, C)", "تلقائي في fit"),
                   ("الهدف", "فئات 0..K−1 كأعداد صحيحة مع sparse CCE", "`y.astype('int64')`"), ("التقسيم", "تدريب/تحقق/اختبار بنسب فئات متقاربة", "300 / 150 / 150"), ("التحقق من الأشكال", "قبل fit دائمًا", "`print(X.shape, X.dtype, y.shape, np.unique(y))`")],
                  ["rtl", "rtl", "code"])
    practical_note("مع صور حقيقية بحجوم مختلفة أضف خطوة **إعادة التحجيم** إلى حجم موحّد (`tf.image.resize` أو `layers.Resizing`)، وللألوان C = 3. الرسوم البيانية للأسعار (شموع/خطوط) يمكن تحويلها إلى صور رمادية ثابتة الحجم بنفس الطريقة.")
    h2("2) البنية: الأسئلة الأحد عشر", "2) Architecture: the eleven questions")
    compare_table(["السؤال", "الجواب"],
                  [("لماذا CNN؟", "بيانات صور: الجوار والإزاحة مهمان (الأسبوع 08)"), ("المدخل", "(16, 16, 1) في [0, 1]"), ("الهدف", "3 فئات متنافية كأعداد صحيحة"), ("البنية", "كتلتان Conv/Pool ثم Dense 32 ثم Dense 3"), ("الأشكال", "16×16×8 → 8×8×8 → 8×8×16 → 4×4×16 → 256 → 32 → 3"),
                   ("التنشيط", "ReLU في Conv/Dense؛ softmax للمخرج"), ("الخسارة", "sparse_categorical_crossentropy"), ("المقياس", "accuracy (فئات متوازنة) + مصفوفة التباس"), ("المحسّن", "Adam 2e-3"), ("الدفعة", "32 → 10 تحديثات/حقبة"), ("الحقب", "8 (المسألة سهلة؛ راقب val)")],
                  ["rtl", "rtl"])
    st.code(CODE, language="python")
    h2("3) التدريب و4) التقييم", "3) Training & 4) Evaluation")
    epochs = st.slider("epochs", 2, 12, 8, key="w09_ep")
    r = cnn_shapes_run(epochs=int(epochs))
    with st.expander("model.summary()", icon=":material/table_view:"):
        st.code(r["summary"].strip(), language="text")
    h_ = r["history"]; e = np.arange(1, len(h_["loss"]) + 1)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=e, y=h_["loss"], name="loss", line=dict(color="#2F6FB5", width=2)))
        fig.add_trace(go.Scatter(x=e, y=h_["val_loss"], name="val_loss", line=dict(color="#C8473A", width=3)))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch", plot_bgcolor="#FFFDF9", paper_bgcolor="#FFFDF9", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch", key="w09_loss")
    with c2:
        cm = r["cm"]
        table(["حقيقي \\ متنبأ"] + r["classes"], [(r["classes"][i],) + tuple(str(v) for v in row) for i, row in enumerate(cm)], ["rtl", "num", "num", "num"])
        st.markdown(f"**test accuracy = {r['test_acc']:.3f}** · test loss = {r['test_loss']:.3f} · المعلمات {r['n_params']:,}")
    intuition("المسألة سهلة عمدًا: الدقة تقترب من 1 في حقب قليلة، ومصفوفة التباس قطرية. المهم هنا **المنهج والأشكال**؛ الدرس التالي يجعل المسألة صعبة (بيانات قليلة) لنرى فرط التخصيص وعلاجه.")
    with st.container(horizontal=True):
        st.button("حاسبة الأشكال لهذه الشبكة", icon=":material/science:", on_click=goto, args=("labs.cnn_shape_calculator",), key="w09_lab_calc")
        st.button("المكافئ في PyTorch (nn.Conv2d)", icon=":material/swap_horiz:", on_click=goto, args=("foundations.frameworks.concept_mapping",), key="w09_go_pt")
    common_mistake("تمرير y بـ one-hot مع `sparse_categorical_crossentropy` أو العكس: أخطاء الرتبة من الوحدة 21. هنا y أعداد صحيحة → sparse. وللمخرج 3 وحدات softmax لا وحدة واحدة.")
    quiz("w09.build", [
        Q("شكل X الصحيح لـ 600 صورة رمادية 16×16:", ["(600, 16, 16)", "(600, 16, 16, 1)", "(16, 16, 600)"], 1, ""),
        Q("Dense(32) بعد Flatten لـ 4×4×16: المعلمات", ["256", "8,224", "512"], 1, "256×32+32."),
        Q("الخسارة لثلاث فئات بأعداد صحيحة:", ["binary_crossentropy", "sparse_categorical_crossentropy", "mse"], 1, ""),
        Q("مصفوفة التباس القطرية تعني…", ["أخطاء كثيرة", "تصنيفًا صحيحًا لكل الفئات", "فرط تخصيص"], 1, ""),
    ])
    takeaway("صور → [0,1] + قناة + دفعة؛ y أعداد صحيحة. CNN صغيرة بأسئلتها الأحد عشر. summary قبل fit، مصفوفة التباس بعد evaluate.")
    lesson_footer(LESSON, ["المعالجة المسبقة كجدول.", "البنية والأسئلة.", "التدريب والتقييم الحيان."])
