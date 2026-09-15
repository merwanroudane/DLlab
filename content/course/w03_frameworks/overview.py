import streamlit as st

from components.callouts import practical_note, takeaway
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
    objectives_ar=["خريطة الأسبوع وربطه بصفحات الوحدة 21.", "اختبار قبلي على الموترات وKeras/TensorFlow."],
    terms=["tensor", "shape", "dtype"],
    difficulty="beginner",
    summary_ar="Keras واجهة عليا فوق TensorFlow؛ مراجعة الموتر (shape/rank/dtype/device)؛ أول شبكة بتفكيك كامل؛ ما تحت fit؛ المكافئ في PyTorch.",
)


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["Keras / TensorFlow roles", "Tensor review", "First network (all outputs)", "Under fit(): GradientTape · tf.data · devices", "PyTorch equivalent", "Post-test"], active=0)
    table(["الموضوع", "درس الأسبوع", "صفحات الوحدة 21 التي يعتمد عليها"],
          [("Keras واجهة عليا؛ TensorFlow منظومة؛ العلاقة في بيئة المقرر", "النظرة العامة + أول شبكة", "خريطة المنظومة · ما هي Keras · ما هو TensorFlow"), ("الموتر: shape / rank / dtype / device", "أول شبكة (مراجعة)", "tf.Tensor + مستكشف الموترات"),
           ("Layers، Model، Sequential، Functional، summary", "أول شبكة", "الطبقات والنماذج · تفكيك summary"), ("compile وfit بكل معاملاتهما، History، evaluate، predict، callbacks", "أول شبكة", "compile · fit · evaluate/predict · الاستدعاءات · مستكشف المخرجات"),
           ("GradientTape، tf.data، الأجهزة، اللقطات", "تحت الغطاء", "GradientTape · tf.data · TensorBoard · المعرض (الوحدة 22)"), ("PyTorch: nn.Module + حلقة، والأخطاء الخمسة", "المكافئ في PyTorch", "PyTorch · حلقة التدريب · الأخطاء")],
          ["rtl", "rtl", "rtl"])
    practical_note("الوحدة 21 هي المرجع التفصيلي؛ هذا الأسبوع يجمعها في **مسار عمل واحد** على مسألة اقتصادية واحدة، ويضيف زر «المكافئ في PyTorch» عند كل خطوة.")
    with st.container(horizontal=True):
        st.button("الوحدة 21 — أطر العمل", icon=":material/deployed_code:", on_click=go, args=("foundations.frameworks",), key="w03_go_fw")
        st.button("الوحدة 22 — المعرض", icon=":material/photo_library:", on_click=go, args=("foundations.gallery",), key="w03_go_gal")
    h2("الاختبار القبلي", "Pre-test")
    quiz("w03.pretest", [
        Q("Keras في هذا المقرر…", ["إطار مستقل", "واجهة عليا تعمل فوق TensorFlow", "بيئة تشغيل"], 1, ""),
        Q("`Input(shape=(12,))` يعني…", ["12 ملاحظة", "كل ملاحظة 12 خاصية", "12 طبقة"], 1, ""),
        Q("`compile()`…", ["يدرّب", "يربط المحسّن والخسارة والمقاييس", "يحفظ"], 1, ""),
        Q("`history.history['val_loss']` طولها", ["عدد الدفعات", "عدد الحقب", "عدد الملاحظات"], 1, ""),
        Q("في PyTorch، ما يقابل fit هو…", ["compile", "حلقة تدريب تكتبها", "summary"], 1, ""),
    ], title_ar="الاختبار القبلي — الأسبوع 03")
    takeaway("الأسبوع 03 = الوحدة 21 كمسار عمل واحد: أول شبكة كاملة في Keras، ما تحتها في TensorFlow، ومكافئها في PyTorch.")
    lesson_footer(LESSON, ["الخريطة والروابط.", "الاختبار القبلي."])
