import streamlit as st

from components.callouts import intuition, takeaway
from components.gallery import Frame6, gallery_item, notebook_cell, svg_line_chart
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from labs.fw import gallery_outputs, keras_tensorboard_run

LESSON = Lesson(
    id="foundations.gallery.keras_tf",
    title_ar="معرض TensorFlow وKeras: الموتر، summary، سجل fit، منحنى History، TensorBoard",
    title_en="TensorFlow & Keras Gallery: Tensor, summary, fit log, History plot, TensorBoard",
    module="foundations.gallery",
    order=2,
    prerequisites=["foundations.gallery.orientation", "foundations.frameworks.keras.output_explorer"],
    objectives_ar=["قراءة مخرج tf.Tensor كما يظهر في خلية.", "قراءة model.summary() وسجل fit وevaluate/predict كما تظهر فعليًا.", "قراءة رسم History ولوحة Scalars في TensorBoard."],
    terms=["tensor", "epoch", "loss"],
    difficulty="beginner",
    summary_ar="ست لقطات حقيقية (من النسخ المثبتة) في إطار دفتر: موتر، ملخص، سجل، evaluate/predict، رسم، TensorBoard — كل واحدة بالأسئلة الستة.",
)


def _history_plot_html(h: dict) -> str:
    return svg_line_chart([("loss", h["loss"], "#1f77b4"), ("val_loss", h["val_loss"], "#ff7f0e")], title="model loss", start_at=0)


def render() -> None:
    lesson_header(LESSON)
    o = gallery_outputs()
    st.caption("كل مخرج أدناه أنتجته النسخ المثبتة في هذا المشروع لحظة فتح الصفحة؛ الإطار المحيط (خلية/طرفية) إعادة بناء تعليمية.")
    h2("1) مخرج موتر TensorFlow", "1) TensorFlow tensor output")
    gallery_item("موتر TensorFlow في خلية", "TensorFlow tensor output",
                 notebook_cell("import tensorflow as tf\nt = tf.constant([[1., 2., 3.], [4., 5., 6.]])\nt", o["tf_tensor"].split("\n>>> t\n")[1].split("\n>>> t.shape")[0], n=1),
                 Frame6(where_ar="خلية دفتر بعد استيراد TensorFlow؛ لا نموذج بعد.", what_ar="تعبير أخير `t` فيعرض الدفتر `repr` الموتر.",
                        code_where_ar="ثلاثة أسطر: استيراد، إنشاء ثابت، عرض.", output_where_ar="`<tf.Tensor: shape=(2, 3), dtype=float32, numpy=array([...])>` — كتلة واحدة.",
                        numbers_ar="`shape=(2, 3)` صفان × ثلاثة أعمدة؛ `dtype=float32` لأن القيم كُتبت بفاصلة عشرية؛ `numpy=` القيم كما تعرضها NumPy.",
                        notice_ar="الثلاثي (shape, dtype, values) يظهر في كل repr — اقرأه دائمًا قبل القيم. لو كانت القيم بلا فاصلة لكان dtype=int32.",
                        extra_ar=["`.device` لا يظهر في repr؛ اسأل عنه صراحةً."]))
    h2("2) model.summary()", "2) model.summary()")
    gallery_item("ملخص نموذج Keras", "Keras model.summary() output",
                 notebook_cell('model = keras.Sequential([\n    layers.Input(shape=(2,)),\n    layers.Dense(16, activation="relu"),\n    layers.Dense(8, activation="relu"),\n    layers.Dense(1, activation="sigmoid"),\n], name="moons_mlp")\nmodel.summary()', o["keras_summary"].strip(), n=2),
                 Frame6(where_ar="بعد بناء النموذج وقبل compile/fit.", what_ar="جدول بإطار من خطوط: ثلاثة أعمدة وثلاثة صفوف (طبقة لكل صف) وثلاثة أسطر إجمالية.",
                        code_where_ar="Sequential بثلاث طبقات كثيفة و`summary()` في آخر سطر (يطبع، لا يعيد قيمة → لا `Out`).", output_where_ar="الجدول كاملًا هو مخرج مطبوع (stdout).",
                        numbers_ar="`(None, 16)`: None بُعد الدفعة. `48 = 2×16+16`، `136 = 16×8+8`، `9 = 8×1+1`. `Total params: 193 (772.00 B)` = 193 × 4 بايت.",
                        notice_ar="Trainable = Total هنا (لا BN ولا تجميد). لو اختلفا فابحث عن السبب. الأسماء `dense`, `dense_1` تلقائية وتتغير عند إعادة التشغيل.",
                        extra_ar=["تفكيك كامل في درس «تفكيك summary» بالوحدة 21."]))
    h2("3) سجل fit()", "3) fit() log")
    gallery_item("سجل تدريب Keras (verbose=2)", "Keras fit() log",
                 notebook_cell('model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=["accuracy"])\nhistory = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32, verbose=2)', o["keras_fit_log"].strip(), n=3),
                 Frame6(where_ar="أثناء التدريب: هذا النص يظهر تدريجيًا، سطران لكل حقبة.", what_ar="`Epoch k/5` ثم سطر بالدفعات والزمن وأربعة مقاييس.",
                        code_where_ar="compile ثم fit بـ `verbose=2`. مع `verbose=1` يظهر شريط تقدم متحرك `━━━━` بدل السطر الثابت.", output_where_ar="10 أسطر مطبوعة؛ و`history` كائن أُسند فلا يظهر.",
                        numbers_ar="`15/15` دفعات (450/32 = 14.06 → 15)؛ `1s - 50ms/step` زمن الحقبة والخطوة (الأولى أبطأ)؛ accuracy/loss تراكمية على التدريب؛ val_* على التحقق كاملًا بعد الحقبة.",
                        notice_ar="loss ينخفض كل حقبة، val_loss ينخفض ثم يستقر ≈ 0.35 من الحقبة 3: بداية ثبات — مع حقب أكثر قد يصعد (فرط تخصيص) وهنا يعمل الإيقاف المبكر.",
                        extra_ar=["الترتيب أبجدي (accuracy قبل loss) لا بحسب الأهمية.", "الحقبة الأولى أبطأ بسبب بناء الرسم/الترجمة."]))
    h2("4) evaluate() و predict()", "4) evaluate() & predict()")
    ev_code, ev_out = o["keras_evaluate"].split("\n", 1)
    gallery_item("evaluate وpredict في خلية", "evaluate / predict output", notebook_cell(ev_code.replace(">>> ", "") + "\n" + o["keras_evaluate"].split(">>> ")[2].split("\n")[0], ev_out.replace(">>> model.predict(X_val[:3], verbose=0)\n", "\n"), n=4),
                 Frame6(where_ar="بعد التدريب.", what_ar="قائمة من رقمين، ثم مصفوفة (3, 1).", code_where_ar="سطران: evaluate على التحقق، predict على 3 ملاحظات.", output_where_ar="القائمة `[loss, accuracy]` ثم `array([[…]], dtype=float32)`.",
                        numbers_ar="0.353 خسارة التحقق (تطابق val_loss لآخر حقبة)، 0.833 دقة. الاحتمالات 0.96/0.82/0.93 → كلها فئة 1 عند عتبة 0.5.",
                        notice_ar="predict يعطي احتمالات في عمود واحد لأن المخرج `Dense(1, sigmoid)`؛ الفئة قرارك بالعتبة.", extra_ar=[]))
    h2("5) رسم History", "5) Training-history plot")
    gallery_item("منحنيا التدريب والتحقق من History", "Keras training-history plot",
                 notebook_cell('import matplotlib.pyplot as plt\nplt.plot(history.history["loss"], label="loss")\nplt.plot(history.history["val_loss"], label="val_loss")\nplt.xlabel("epoch"); plt.ylabel("loss"); plt.title("model loss"); plt.legend(); plt.show()', "", n=5, image_html=_history_plot_html(o["keras_history"])),
                 Frame6(where_ar="بعد fit: نرسم القاموس `history.history`.", what_ar="رسم بمنحنيين: التدريب (أزرق) والتحقق (برتقالي) عبر الحقب.",
                        code_where_ar="matplotlib على القوائم (هنا مُعاد بناؤه كرسم SVG بنفس البيانات).", output_where_ar="الرسم يظهر تحت الخلية مباشرة (بلا Out لأن plt.show لا يعيد قيمة).",
                        numbers_ar="المحور الأفقي حقب (من 0 في matplotlib، من 1 هنا)؛ الرأسي الخسارة. القيم هي نفسها أرقام السجل.",
                        notice_ar="التقارب بين المنحنيين = تعميم جيد؛ افتراقهما (val فوق train ويصعد) = فرط تخصيص (وحدة 19). ابحث عن حقبة الحد الأدنى لـ val_loss.", extra_ar=[]))
    h2("6) TensorBoard — لوحة Scalars", "6) TensorBoard Scalars view")
    tb = keras_tensorboard_run(12)
    panel = svg_line_chart([("train", tb["history"]["loss"], "#E8710A"), ("validation", tb["history"]["val_loss"], "#12B5CB")], width=440, height=220, xlabel="step", ylabel="")
    tb_html = ('<div style="display:flex;justify-content:space-between;align-items:center;background:#2B2A28;color:#F1EFEA;padding:.3rem .7rem;border-radius:10px 10px 0 0;font-size:.8rem"><span class="en">TensorBoard · Scalars (localhost:6006)</span>'
               '<span style="background:#D9A21B;color:#2B2A28;border-radius:6px;padding:.1rem .5rem;font-weight:600">Educational recreation · إعادة بناء تعليمية</span></div>'
               '<div dir="ltr" style="text-align:left;display:flex;border:1px solid #B9B2A6;border-top:0;border-radius:0 0 10px 10px;background:#FFF;font-family:Inter,Segoe UI,sans-serif;font-size:12px">'
               '<div style="width:160px;border-right:1px solid #E5E0D6;padding:8px"><div style="font-weight:600;margin-bottom:6px">Runs</div>'
               '<div>☑ <span style="color:#E8710A">■</span> run1/train</div><div>☑ <span style="color:#12B5CB">■</span> run1/validation</div>'
               '<div style="margin-top:10px;font-weight:600">Smoothing</div><div>0.6 ────●──</div></div>'
               f'<div style="flex:1;padding:8px"><div style="font-weight:600;margin-bottom:4px">epoch_loss</div>{panel}<div style="color:#6B675F">epoch_accuracy · epoch_learning_rate ▸</div></div></div>')
    gallery_item("TensorBoard: لوحة Scalars", "TensorBoard Scalars view", tb_html,
                 Frame6(where_ar="في المتصفح على `localhost:6006` بعد `tensorboard --logdir logs`؛ التدريب كتب السجلات باستدعاء `TensorBoard(log_dir)`.",
                        what_ar="يسار: قائمة التشغيلات (run1/train وrun1/validation بلونين) وشريط التنعيم. يمين: بطاقة `epoch_loss` وبطاقات أخرى مطوية.",
                        code_where_ar="ليس هنا — في الدفتر/الملف الذي شغّل fit مع الاستدعاء. TensorBoard يقرأ الملفات فقط.", output_where_ar="المنحنيات نفسها؛ كل نقطة = قيمة كُتبت في نهاية حقبة.",
                        numbers_ar="المحور الأفقي step (= الحقبة هنا)، الرأسي الخسارة. train (برتقالي) وvalidation (أزرق) من مجلدين.",
                        notice_ar="اضبط Smoothing إلى 0 لرؤية القيم الحقيقية. قارن تشغيلات متعددة (run1, run2) في نفس البطاقة لمقارنة معلمات فائقة.", extra_ar=["شرح كل لوحة في درس TensorBoard بالوحدة 21."]))
    intuition("ست لقطات، وجميعها نص أو رسم أنتجه سطر كود يمكنك الإشارة إليه. لا شيء «يظهر من تلقاء نفسه».")
    quiz("gallery.keras", [
        Q("في repr موتر TensorFlow، الثلاثي الذي تقرؤه أولًا:", ["القيم فقط", "shape وdtype ثم القيم", "الجهاز"], 1, "الترتيب."),
        Q("`15/15` في سطر fit:", ["15 حقبة", "15 دفعة في الحقبة", "15%"], 1, "⌈n/B⌉."),
        Q("`model.summary()` في الدفتر لا يُنتج `Out` لأنه…", ["يفشل", "يطبع ولا يعيد قيمة", "يعيد None سرًا"], 1, "stdout."),
        Q("TensorBoard يعرض ما…", ["يحسبه بنفسه", "كُتب في ملفات السجلات أثناء التدريب", "يوجد في الدفتر"], 1, "يقرأ."),
    ])
    takeaway("موتر: shape/dtype/values. summary: طبقات ومعلمات. fit: سطران لكل حقبة بثمانية أرقام. evaluate: [loss, metrics]. predict: احتمالات. History: منحنيان. TensorBoard: نفس الأرقام في المتصفح.")
    lesson_footer(LESSON, ["ست لقطات من مخرجات حقيقية.", "الأسئلة الستة لكل واحدة.", "ما يجب ملاحظته في كل سطح."])
