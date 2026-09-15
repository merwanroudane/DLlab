import streamlit as st

from components.callouts import practical_note, takeaway, why
from components.lesson_layout import h2, lesson_footer, lesson_header
from core.models import Lesson
from core.registry import get_registry
from core.routing import go
from core.rtl import table

LESSON = Lesson(
    id="foundations.start.learning_roadmap",
    title_ar="خريطة المقرر ومسار التعلم",
    title_en="Course Map & Learning Roadmap",
    module="foundations.start",
    order=2,
    prerequisites=["foundations.start.how_to_use"],
    objectives_ar=[
        "رؤية الصورة الكاملة: من البيانات إلى الشبكات التكرارية.",
        "معرفة المسار المقترح لثلاث فئات من المتعلمين.",
        "ربط المحاور السبعة الرسمية بالأسابيع الخمسة عشر.",
    ],
    difficulty="beginner",
    summary_ar="الصورة الكاملة للمقرر: وحدات الأسس، المحاور الرسمية، والأسابيع، مع مسارات مقترحة.",
)

FOUNDATION_MODULES = [
    ("0", "ابدأ من هنا", "Start Here"),
    ("1", "أسس البيانات", "Data Foundations"),
    ("2", "بايثون للتعلم العميق", "Python Foundations"),
    ("3", "الأسس الرياضية", "Mathematical Foundations"),
    ("4", "الجبر الخطي", "Linear Algebra"),
    ("5", "التفاضل للتعلم العميق", "Calculus for DL"),
    ("6", "الاحتمال والإحصاء", "Probability & Statistics"),
    ("7", "أسس التعلم الآلي", "Machine Learning Foundations"),
    ("8", "إعداد البيانات", "Data Preparation"),
    ("9", "من الانحدار إلى الخلية العصبية", "From Regression to Neuron"),
    ("10", "بنية الشبكة العصبية", "Neural Network Architecture"),
    ("11", "أسس دوال التنشيط", "Activation Foundations"),
    ("12", "التمرير الأمامي", "Forward Pass"),
    ("13", "الخطأ والخسارة والهدف", "Error, Loss & Objective"),
    ("14", "الانتشار الخلفي", "Backpropagation"),
    ("15", "التحسين", "Optimization"),
    ("16", "حلقة التدريب", "Training Loop"),
    ("17", "الدفعة والحقبة والتكرار", "Batch / Epoch / Iteration"),
    ("18", "تقييم النموذج", "Model Evaluation"),
    ("19", "التعميم", "Generalization"),
    ("20", "التنظيم", "Regularization"),
    ("21", "أكاديمية أطر العمل: Keras وTensorFlow وPyTorch", "Frameworks Academy"),
    ("22", "معرض أطر العمل البصري", "Framework Visual Gallery"),
    ("23", "جاهز للتعلم العميق", "Ready for Deep Learning"),
]

AXES = [
    ("1", "أساسيات التعلم الآلي", "الأسبوعان 01–02"),
    ("2", "الشبكات العصبية الأمامية", "الأسابيع 03–04 و07"),
    ("3", "خوارزميات التحسين", "الأسبوعان 05–06"),
    ("4", "الشبكات العصبية الالتفافية CNN", "الأسبوعان 08–09"),
    ("5", "الشبكات العصبية التكرارية RNN", "الأسبوع 10"),
    ("6", "شبكات LSTM", "الأسبوع 11"),
    ("7", "شبكات GRU", "الأسبوع 12 (ثم GPU في 13 والمشروع في 14–15)"),
]


def render() -> None:
    lesson_header(LESSON)
    reg = get_registry()

    h2("الصورة الكاملة", "The big picture")
    st.markdown(
        """
المنصة مبنية على طبقتين:

1. **أكاديمية الأسس** — 24 وحدة تبني المعرفة القبلية درسًا درسًا.
2. **المقرر الرسمي** — 15 أسبوعًا تطبّق تلك المعرفة على الشبكات العصبية الأمامية والالتفافية والتكرارية.

القاعدة: **لا يظهر مصطلح قبل شرحه**. لذلك إذا فتحت أسبوعًا من المقرر ووجدت مفهومًا غير واضح، ستجد
في أعلى الدرس رابطًا إلى الدرس التأسيسي الذي يشرحه.
"""
    )

    h2("وحدات أكاديمية الأسس", "Foundations modules")
    available = {m.title_en for m in reg.modules_of("foundations")}
    rows = []
    for n, ar, en in FOUNDATION_MODULES:
        status = "متاحة الآن" if any(en in a for a in available) else "قيد البناء ضمن الخطة"
        rows.append((n, ar, en, status))
    table(["#", "الوحدة", "Module", "الحالة"], rows, ["num", "rtl", "ltr", "rtl"])
    practical_note(
        "الوحدات تُنشر تباعًا حسب خطة التنفيذ. الوحدات المتاحة تظهر في اللوحة اليمنى، "
        "والوحدات المخطط لها لا تظهر كصفحات فارغة أبدًا."
    )

    h2("المحاور الرسمية السبعة وربطها بالأسابيع", "Official axes → weeks")
    table(["#", "المحور", "الأسابيع"], AXES, ["num", "rtl", "rtl"])

    h2("مسارات مقترحة", "Suggested paths")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**🟢 مبتدئ تمامًا**")
            st.markdown("الوحدات 0 → 23 بالترتيب، ثم الأسابيع 01 → 15. لا تتخطَّ الرياضيات.")
    with c2:
        with st.container(border=True):
            st.markdown("**🟡 يعرف الإحصاء وبايثون**")
            st.markdown("ابدأ بالفحص الذاتي، ثم الوحدات 4–6 (جبر، تفاضل، احتمال)، ثم 9 فصاعدًا.")
    with c3:
        with st.container(border=True):
            st.markdown("**🔵 يعرف التعلم الآلي**")
            st.markdown("الوحدات 9–21 ثم المقرر من الأسبوع 03. راجع الوحدة 17 (الدفعة/الحقبة) حتى لو ظننت أنك تعرفها.")
    why(
        "لماذا نصرّ على الترتيب؟ لأن مفهومًا مثل «التدرج» يعتمد على «المشتقة الجزئية» التي تعتمد على "
        "«المشتقة» التي تعتمد على «الميل». القفز فوق حلقة واحدة يجعل الانتشار الخلفي طلاسم."
    )
    with st.container(horizontal=True):
        st.button("افتح خريطة المعرفة التفاعلية", icon=":material/account_tree:", on_click=go, args=("map",),
                  key="roadmap_map")
        st.button("ابدأ الفحص الذاتي", icon=":material/checklist:", on_click=go,
                  args=("foundations.start.self_check",), key="roadmap_check")
    takeaway("طبقتان: أسس ثم مقرر. سبعة محاور رسمية موزعة على 15 أسبوعًا. اختر مسارك ولا تقفز فوق الحلقات.")
    lesson_footer(LESSON, [
        "أكاديمية الأسس = 24 وحدة تبني المعرفة القبلية.",
        "المقرر الرسمي = 15 أسبوعًا إلزاميًا مربوطًا بالأسس.",
        "المحاور السبعة: ML، FFNN، التحسين، CNN، RNN، LSTM، GRU.",
    ])
