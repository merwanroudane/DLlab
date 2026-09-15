import streamlit as st

from components.callouts import common_mistake, intuition, practical_note, takeaway, why
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.rtl import table
from labs.fw import versions

LESSON = Lesson(
    id="foundations.frameworks.concept_mapping",
    title_ar="خريطة المفاهيم: Keras/TensorFlow مقابل PyTorch — جدول تفاعلي",
    title_en="Concept Mapping: Keras/TensorFlow vs PyTorch — Interactive Table",
    module="foundations.frameworks",
    order=28,
    prerequisites=["foundations.frameworks.pytorch.saving_errors", "foundations.frameworks.keras.code_lab_standard"],
    objectives_ar=["ربط كل مفهوم (موتر، نموذج، طبقة، خسارة، محسّن، تدريب، اشتقاق، بيانات، وضع، حفظ) بواجهته في الإطارين.", "البحث والتصفية في الجدول بحسب الفئة أو الكلمة.", "الوعي بأن الواجهات تتغير مع النسخ وأن المرجع هو النسخة المثبتة في المشروع."],
    terms=["tensor", "model", "loss", "optimizer"],
    difficulty="intermediate",
    summary_ar="مفهوم واحد، واجهتان. الجدول يربط 30+ مفهومًا بمقابله في Keras/TensorFlow وPyTorch مع ملاحظة الفرق. النسخ المثبتة هي المرجع.",
)

ROWS = [
    ("الموترات", "موتر", "`tf.Tensor` / `tf.constant`", "`torch.Tensor` / `torch.tensor`", "المفهوم واحد؛ int32 مقابل int64 افتراضيًا"),
    ("الموترات", "معلمة قابلة للاشتقاق", "`tf.Variable`", "`requires_grad=True` / `nn.Parameter`", "PyTorch: علامة على الموتر"),
    ("الموترات", "الجهاز", "تلقائي؛ `with tf.device('/GPU:0')`", "`x.to(device)`, `model.to(device)`", "PyTorch صريح دائمًا"),
    ("الموترات", "إلى NumPy", "`x.numpy()`", "`x.detach().cpu().numpy()`", "PyTorch يحتاج فصل ونقل"),
    ("الموترات", "إضافة محور", "`tf.expand_dims(x, 1)`", "`x.unsqueeze(1)`", "—"),
    ("الموترات", "اختزال على محور", "`tf.reduce_mean(x, axis=0)`", "`x.mean(dim=0)`", "axis مقابل dim"),
    ("النموذج", "نموذج", "`keras.Model` / `keras.Sequential`", "`nn.Module` (صنف) / `nn.Sequential`", "Keras: قائمة أولًا؛ PyTorch: صنف أولًا"),
    ("النموذج", "التمرير الأمامي", "ضمني (`model(x)`)", "`def forward(self, x)` ثم `model(x)`", "PyTorch تكتبه"),
    ("النموذج", "طبقة كثيفة", "`keras.layers.Dense(units)`", "`nn.Linear(in, out)`", "PyTorch يحتاج in صريحًا؛ الوزن (out, in)"),
    ("النموذج", "تنشيط", "`activation='relu'` / `layers.ReLU()`", "`nn.ReLU()` / `F.relu`", "—"),
    ("النموذج", "Dropout / BN", "`layers.Dropout(p)`, `layers.BatchNormalization()`", "`nn.Dropout(p)`, `nn.BatchNorm1d(f)`", "السلوك يعتمد على الوضع في الاثنين"),
    ("النموذج", "التفافية", "`layers.Conv2D(k, (3,3))` (channels last)", "`nn.Conv2d(c_in, k, 3)` (channels first)", "ترتيب المحاور مختلف! (الأسبوع 8)"),
    ("النموذج", "المعلمات", "`model.weights` / `trainable_weights`", "`model.parameters()` / `named_parameters()`", "—"),
    ("النموذج", "الملخص", "`model.summary()`", "`print(model)` + `sum(p.numel())`", "PyTorch بلا أشكال مخرجات"),
    ("التدريب", "الخسارة", "`loss='mse'` / `keras.losses.*`", "`nn.MSELoss()` / `nn.CrossEntropyLoss()` / `F.*`", "PyTorch يأخذ logits افتراضيًا"),
    ("التدريب", "ثنائي", "sigmoid + `binary_crossentropy`", "logits + `BCEWithLogitsLoss`", "الهدف (n,1) float في الاثنين"),
    ("التدريب", "متعدد الفئات", "softmax + `sparse_categorical_crossentropy`", "logits + `CrossEntropyLoss`", "الهدف أعداد صحيحة؛ PyTorch: int64"),
    ("التدريب", "المحسّن", "`keras.optimizers.Adam(learning_rate)`", "`torch.optim.Adam(model.parameters(), lr)`", "PyTorch يستلم المعلمات"),
    ("التدريب", "جدول معدل التعلم", "`callbacks.ReduceLROnPlateau` / `LearningRateScheduler`", "`optim.lr_scheduler.*` + `sched.step()`", "PyTorch: استدعاء يدوي"),
    ("التدريب", "الربط", "`model.compile(optimizer, loss, metrics)`", "لا يوجد: كائنات منفصلة", "—"),
    ("التدريب", "التدريب العالي المستوى", "`model.fit(X, y, epochs, batch_size, validation_data, callbacks)`", "حلقة صريحة (أو مكتبات مثل Lightning — تعميق)", "الفرق الجوهري"),
    ("التدريب", "الاشتقاق التلقائي", "داخل fit؛ `tf.GradientTape` يدويًا", "Autograd: `loss.backward()`", "—"),
    ("التدريب", "تصفير التدرجات", "غير مطلوب", "`optimizer.zero_grad()`", "تراكم في .grad"),
    ("التدريب", "خطوة التحديث", "داخل fit؛ `optimizer.apply_gradients(zip(g, v))`", "`optimizer.step()`", "—"),
    ("التدريب", "الإيقاف المبكر", "`callbacks.EarlyStopping(patience, restore_best_weights)`", "تكتبه: تتبع best_val وpatience ونسخة من state_dict", "—"),
    ("التدريب", "المقاييس", "`metrics=['accuracy']` + History", "تحسبها في الحلقة (أو torchmetrics)", "—"),
    ("التقييم", "وضع الاستدلال", "تلقائي في evaluate/predict", "`model.eval()` + `torch.no_grad()`", "مسؤوليتك"),
    ("التقييم", "التقييم", "`model.evaluate(X, y)`", "حلقة تحقق تكتبها", "—"),
    ("التقييم", "التنبؤ", "`model.predict(X)` → NumPy", "`model(x)` تحت no_grad → موتر", "—"),
    ("البيانات", "خط البيانات", "`tf.data.Dataset` / مصفوفات NumPy", "`Dataset` + `DataLoader`", "نفس الأفكار: خلط قبل تجميع"),
    ("البيانات", "الدفعات", "`batch_size=` في fit أو `.batch()`", "`DataLoader(batch_size=, shuffle=)`", "—"),
    ("الحفظ", "الحفظ", "`model.save('m.keras')` (كامل)", "`torch.save(model.state_dict(), 'm.pt')` (أوزان)", "PyTorch يحتاج الكود للتحميل"),
    ("الحفظ", "التحميل", "`keras.saving.load_model`", "`Net(); load_state_dict(torch.load(...))`", "—"),
    ("الحفظ", "النشر", "`model.export()` SavedModel", "TorchScript / `torch.export` / ONNX", "تعميق"),
    ("الأدوات", "الاستنساخ", "`keras.utils.set_random_seed(0)`", "`torch.manual_seed(0)`", "—"),
    ("الأدوات", "المراقبة", "`callbacks.TensorBoard`", "`torch.utils.tensorboard.SummaryWriter`", "نفس TensorBoard"),
    ("الأدوات", "GPU؟", "`tf.config.list_physical_devices('GPU')`", "`torch.cuda.is_available()`", "—"),
]


def render() -> None:
    lesson_header(LESSON)
    why("بعد المسارين الثلاثة (Keras، TensorFlow، PyTorch) تملك المفاهيم. هذا الجدول يمنعك من تعلّم المفهوم نفسه مرتين: عندما تعرف «الخسارة الثنائية من logits» في PyTorch فأنت تعرفها في Keras — الاسم فقط يختلف.")
    c1, c2 = st.columns([1, 2])
    with c1:
        cats = ["الكل"] + sorted({r[0] for r in ROWS}, key=[r[0] for r in ROWS].index)
        cat = st.selectbox("الفئة", cats, key="cm_cat")
    with c2:
        q = st.text_input("ابحث (عربي أو اسم API)", "", key="cm_q", placeholder="مثل: zero_grad أو الجهاز أو Dense")
    rows = [r for r in ROWS if (cat == "الكل" or r[0] == cat) and (not q or q.lower() in " ".join(r).lower())]
    st.caption(f"{len(rows)} من {len(ROWS)} مفهومًا")
    table(["الفئة", "المفهوم", "Keras / TensorFlow", "PyTorch", "الفرق"], rows, ["rtl", "rtl", "code", "code", "rtl"])
    h2("الأنماط الكبرى", "The big patterns")
    st.markdown("""
1. **الأشياء التي يخفيها fit** (اشتقاق، تصفير، تحديث، وضع، جهاز، مقاييس) هي بالضبط الأسطر التي تكتبها في PyTorch. من يفهم أحدهما يفهم الآخر.
2. **التنشيط الأخير**: Keras يضعه في الطبقة والخسارة تتوقع احتمالات (أو `from_logits=True`)؛ PyTorch يترك logits والخسارة تتولى التنشيط.
3. **أشكال الأوزان**: Keras (in, out)، PyTorch (out, in). الصور: Keras channels-last (N,H,W,C)، PyTorch channels-first (N,C,H,W).
4. **الصراحة**: كل ما هو تلقائي في Keras (جهاز، وضع، تصفير) صريح في PyTorch — وكل ما هو صريح في PyTorch يمكن فعله في TensorFlow بالنزول تحت Keras.
""")
    intuition("اقرأ عمودي الجدول كترجمة بين لغتين تصفان نفس الواقع الرياضي (الوحدات 9–20). المترجم الجيد يفهم المعنى لا الكلمات.")
    h2("النسخ: المرجع هو المثبّت", "Versions: the pinned install is the reference")
    v = versions()
    st.code("\n".join(f"{k:<12} {val}" for k, val in v.items()), language="text")
    practical_note("الواجهات تتغير: Keras 2 → 3 غيّرت الحفظ (`.h5` → `.keras`) وطريقة الاستيراد؛ PyTorch أضاف `inference_mode` و`torch.compile`؛ TensorFlow 1 → 2 قلب نموذج التنفيذ. كل كود في هذه المنصة مُختبَر على النسخ أعلاه المثبتة في `requirements.txt`. عند قراءة مصدر خارجي: اسأل أولًا «أي نسخة؟».")
    common_mistake("ترجمة حرفية بلا فهم: `model.fit` لا يوجد في PyTorch، و`zero_grad` لا معنى له في Keras، و`Dense(64)` بلا `in` لا يعمل في PyTorch. الجدول يترجم **المفاهيم** — التركيب يُعاد بناؤه.")
    quiz("fw.map2", [
        Q("`nn.Linear` يقابل…", ["`tf.Variable`", "`keras.layers.Dense`", "`model.fit`"], 1, "طبقة كثيفة."),
        Q("`optimizer.zero_grad()` يقابل في Keras…", ["`compile`", "لا شيء: fit يتولاه", "`EarlyStopping`"], 1, "تلقائي."),
        Q("Conv2D في Keras يتوقع الصور بترتيب…", ["(N,C,H,W)", "(N,H,W,C)", "(H,W)"], 1, "channels-last."),
        Q("عند اختلاف كود مرجع خارجي عن الجدول، تحقق أولًا من…", ["الجهاز", "النسخة", "البذرة"], 1, "الواجهات تتغير."),
    ])
    takeaway("مفهوم واحد، واجهتان. ما يخفيه fit تكتبه في PyTorch. logits مقابل تنشيط، (in,out) مقابل (out,in)، channels-last مقابل first. المرجع النسخة المثبتة.")
    lesson_footer(LESSON, ["جدول 37 مفهومًا قابل للتصفية.", "الأنماط الأربعة.", "النسخ."])
