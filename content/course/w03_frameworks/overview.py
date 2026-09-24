import streamlit as st

from components import svgkit as K
from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, practical_note, takeaway
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline, table

LESSON = Lesson(
    id="course.w03.overview",
    title_ar="نظرة عامة على الأسبوع 03 والاختبار القبلي",
    title_en="Week 03 Overview & Pre-test",
    module="course.w03",
    order=1,
    prerequisites=["course.w02.ml_to_dl", "foundations.frameworks.ecosystem_map"],
    objectives_ar=["فهم طبقات المنظومة: Python ← NumPy ← Keras ← TensorFlow/PyTorch ← CPU/GPU، بتحريك.", "معرفة ما يقدمه الإطار ولا تقدمه NumPy (اشتقاق آلي، GPU، طبقات جاهزة).",
                   "خريطة الأسبوع وربطه بصفحات الوحدة 21.", "اختبار قبلي على الموترات وKeras/TensorFlow."],
    terms=["tensor", "shape", "dtype", "framework", "keras", "tensorflow", "pytorch", "autodiff", "gpu"],
    difficulty="beginner",
    summary_ar="Keras واجهة عليا فوق TensorFlow؛ الإطار = موترات + اشتقاق آلي + أجهزة + طبقات جاهزة؛ مراجعة الموتر؛ أول شبكة بتفكيك كامل؛ ما تحت fit؛ المكافئ في PyTorch.",
)

_LAYERS = [
    ("Python", "لغة البرنامج: الحلقات والدوال والملفات.", K.BLUE),
    ("NumPy", "مصفوفات وعمليات متجهة سريعة — لكن **بلا اشتقاق آلي وبلا GPU**. كتبنا به حلقة الأسبوع 01 يدويًا.", K.CYAN),
    ("TensorFlow / PyTorch", "**موترات** مثل NumPy + **اشتقاق آلي** (تدرجات بلا حساب يدوي) + تشغيل على **GPU** + أدوات بيانات وحفظ.", K.EMERALD),
    ("Keras", "واجهة **عليا**: طبقات جاهزة، `compile`، `fit`، callbacks. تعمل فوق TensorFlow (خلفيتنا في المقرر).", K.VIOLET),
    ("Your model", "ما تكتبه أنت: بضعة أسطر Keras تستدعي كل الطبقات تحتها.", K.PINK),
    ("CPU / GPU", "العتاد: نفس الكود يعمل على المعالج أو على بطاقة الرسوم دون تغيير (Keras يختار تلقائيًا).", K.AMBER),
]


def _stack_svg(k: int) -> str:
    s = K.svg_open(680, 300)
    order = [5, 0, 1, 2, 3, 4]      # draw bottom (hardware) to top (your model)
    for pos, i in enumerate(order):
        name, _, col = _LAYERS[i]
        y = 250 - pos * 40
        on = i == k
        s += K.box(140, y, 400, 34, name, col, filled=on, size=13)
    s += K.text(600, 60, "high level", size=11, color=K.MUTED)
    s += K.text(600, 270, "low level", size=11, color=K.MUTED)
    s += K.arrow(600, 250, 600, 75, color=K.MUTED)
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["Keras / TensorFlow roles", "Tensor review", "First network (all outputs)", "Under fit(): GradientTape · tf.data · devices", "PyTorch equivalent", "Post-test"], active=0)
    h2("طبقات المنظومة: من يفعل ماذا؟", "The stack: who does what?")
    animation_player("w03_stack", [Frame(_stack_svg(i), caption(f"**{n}**: {t}"), action=n) for i, (n, t, _) in enumerate(_LAYERS)],
                     title_ar="من Python إلى GPU", interval_ms=2400)
    compare_table(["الحاجة", "NumPy وحده", "إطار (TensorFlow/PyTorch)"],
                  [("ضرب مصفوفات", "نعم", "نعم"), ("التدرجات", "تكتبها يدويًا لكل نموذج (الأسبوع 01)", "آليًا لأي نموذج (`GradientTape` / `backward`)"),
                   ("GPU", "لا", "نعم، بلا تغيير في الكود تقريبًا"), ("طبقات جاهزة", "لا", "Dense، Conv2D، LSTM…"), ("حفظ/تحميل وخطوط بيانات", "يدويًا", "مدمجة")],
                  ["rtl", "rtl", "rtl"])
    intuition("كل ما كتبناه يدويًا في الأسبوعين 01–02 (التنبؤ، الخسارة، التدرج، التحديث) موجود في الإطار — لكنه **يُشتق ويُنفّذ آليًا** لأي بنية تبنيها. هذا الأسبوع نرى ذلك، ثم نفتح الغطاء.")
    table(["الموضوع", "درس الأسبوع", "صفحات الوحدة 21 التي يعتمد عليها"],
          [("Keras واجهة عليا؛ TensorFlow منظومة؛ العلاقة في بيئة المقرر", "النظرة العامة + أول شبكة", "خريطة المنظومة · ما هي Keras · ما هو TensorFlow"), ("الموتر: shape / rank / dtype / device", "أول شبكة (مراجعة)", "tf.Tensor + مستكشف الموترات"),
           ("Layers، Model، Sequential، Functional، summary", "أول شبكة", "الطبقات والنماذج · تفكيك summary"), ("compile وfit بكل معاملاتهما، History، evaluate، predict، callbacks", "أول شبكة", "compile · fit · evaluate/predict · الاستدعاءات · مستكشف المخرجات"),
           ("GradientTape، tf.data، الأجهزة، اللقطات", "تحت الغطاء", "GradientTape · tf.data · TensorBoard · المعرض (الوحدة 22)"), ("PyTorch: nn.Module + حلقة، والأخطاء الخمسة", "المكافئ في PyTorch", "PyTorch · حلقة التدريب · الأخطاء")],
          ["rtl", "rtl", "rtl"])
    st.markdown(
        "**في نهاية الأسبوع ستستطيع:**\n"
        "- أن تقرأ أي موتر برتبته وشكله ونوعه وجهازه.\n"
        "- أن تبني شبكة Keras وتحسب معلماتها يدويًا وتقرأ كل مخرج (summary، سجل fit، History، evaluate، predict).\n"
        "- أن تشرح ما يحدث داخل fit: دفعات، GradientTape، apply_gradients.\n"
        "- أن تترجم الشبكة نفسها إلى PyTorch وتتجنب أخطاءه الصامتة."
    )
    practical_note("الوحدة 21 هي المرجع التفصيلي؛ هذا الأسبوع يجمعها في **مسار عمل واحد** على مسألة اقتصادية واحدة، ويضيف زر «المكافئ في PyTorch» عند كل خطوة.")
    with st.container(horizontal=True):
        st.button("الوحدة 21 — أطر العمل", icon=":material/deployed_code:", on_click=go, args=("foundations.frameworks",), key="w03_go_fw")
        st.button("الوحدة 22 — المعرض", icon=":material/photo_library:", on_click=go, args=("foundations.gallery",), key="w03_go_gal")
    h2("الاختبار القبلي", "Pre-test")
    quiz("w03.pretest", [
        Q("Keras في هذا المقرر…", ["إطار مستقل", "واجهة عليا تعمل فوق TensorFlow", "بيئة تشغيل"], 1, "Keras 3 بخلفية TensorFlow."),
        Q("`Input(shape=(12,))` يعني…", ["12 ملاحظة", "كل ملاحظة 12 خاصية", "12 طبقة"], 1, "شكل الملاحظة؛ بُعد الدفعة يُضاف تلقائيًا."),
        Q("`compile()`…", ["يدرّب", "يربط المحسّن والخسارة والمقاييس", "يحفظ"], 1, "التدريب في fit."),
        Q("`history.history['val_loss']` طولها", ["عدد الدفعات", "عدد الحقب", "عدد الملاحظات"], 1, "قيمة لكل حقبة."),
        Q("ما الذي يقدمه الإطار ولا تقدمه NumPy؟", ["ضرب المصفوفات", "الاشتقاق الآلي وGPU", "المصفوفات"], 1, ""),
        Q("في PyTorch، ما يقابل fit هو…", ["compile", "حلقة تدريب تكتبها", "summary"], 1, "حلقة صريحة."),
    ], title_ar="الاختبار القبلي — الأسبوع 03")
    takeaway("الأسبوع 03 = الوحدة 21 كمسار عمل واحد: أول شبكة كاملة في Keras، ما تحتها في TensorFlow، ومكافئها في PyTorch.")
    lesson_footer(LESSON, ["طبقات المنظومة (تحريك).", "الخريطة والروابط.", "الاختبار القبلي."])
