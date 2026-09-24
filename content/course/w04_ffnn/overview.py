import streamlit as st

from components import svgkit as K
from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, practical_note, takeaway
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline, table

LESSON = Lesson(
    id="course.w04.overview",
    title_ar="نظرة عامة على الأسبوع 04 والاختبار القبلي",
    title_en="Week 04 Overview & Pre-test",
    module="course.w04",
    order=1,
    prerequisites=["course.w03.pytorch_equivalent", "foundations.architecture.layers", "foundations.forward.layer_by_layer"],
    objectives_ar=["رؤية كيف تُبنى الشبكة من قطع: عصبون ← طبقة ← شبكة ← رأس مخرج (تحريك).", "خريطة الأسبوع وروابط الأسس.", "اختبار قبلي على التدفق الأمامي والمعلمات."],
    terms=["weight", "bias", "parameter", "neuron", "layer"],
    difficulty="beginner",
    summary_ar="التغذية الأمامية ← MLP ← الطبقات ← الأوزان/الانحيازات ← التمرير الأمامي ← نماذج متعددة الطبقات ← عدّ المعلمات ← التدفق المرئي ← العمق والعرض.",
)

_BLOCKS = [
    ("neuron", "**العصبون**: مجموع موزون + انحياز + تنشيط: `a = f(w·x + b)`. رأيناه في الأسبوعين 01–02 (انحدار خطي ولوجستي).", K.VIOLET),
    ("layer", "**الطبقة**: عدة عصبونات تقرأ نفس المدخل بأوزان مختلفة ⇒ مصفوفة أوزان `W` وعملية واحدة `xW + b`.", K.BLUE),
    ("network", "**الشبكة**: مخرج طبقة = مدخل التالية. الطبقات المخفية تتعلم خصائص جديدة، طبقة بعد طبقة.", K.CYAN),
    ("head", "**رأس المخرج**: تنشيط أخير بحسب المهمة (بلا / sigmoid / softmax) وخسارة تناسبه.", K.PINK),
    ("choose", "**الاختيار**: كم طبقة وكم وحدة؟ نقرر بتجربة على التحقق، لا بالحدس.", K.AMBER),
]


def _blocks_svg(k: int) -> str:
    s = K.svg_open(680, 200)
    if k >= 0:
        s += f'<circle cx="70" cy="100" r="26" fill="{K.VIOLET if k == 0 else K.tint(K.VIOLET, 0.75)}" stroke="{K.VIOLET}" stroke-width="2"/>' + K.text(70, 105, "f", size=14, bold=True, color="#fff" if k == 0 else K.VIOLET)
    if k >= 1:
        for i in range(4):
            s += f'<circle cx="200" cy="{40 + i * 40}" r="16" fill="{K.BLUE if k == 1 else K.tint(K.BLUE, 0.75)}" stroke="{K.BLUE}"/>'
        s += f'<rect x="178" y="18" width="44" height="164" rx="12" fill="none" stroke="{K.BLUE}" stroke-dasharray="4 3"/>'
    if k >= 2:
        for c, xx in enumerate((300, 390)):
            for i in range(4):
                s += f'<circle cx="{xx}" cy="{40 + i * 40}" r="16" fill="{K.CYAN if k == 2 else K.tint(K.CYAN, 0.75)}" stroke="{K.CYAN}"/>'
        for i in range(4):
            for j in range(4):
                s += f'<line x1="216" y1="{40 + i * 40}" x2="284" y2="{40 + j * 40}" stroke="{K.tint(K.CYAN, 0.5)}"/>'
                s += f'<line x1="316" y1="{40 + i * 40}" x2="374" y2="{40 + j * 40}" stroke="{K.tint(K.CYAN, 0.5)}"/>'
    if k >= 3:
        s += K.box(450, 78, 100, 44, "σ / softmax", K.PINK, filled=k == 3, size=12)
        for i in range(4):
            s += f'<line x1="406" y1="{40 + i * 40}" x2="450" y2="100" stroke="{K.tint(K.PINK, 0.5)}"/>'
    if k >= 4:
        s += K.box(575, 78, 95, 44, "depth? width?", K.AMBER, filled=True, size=11)
        s += K.arrow(552, 100, 573, 100, color=K.AMBER)
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["Feedforward concept", "MLP layers", "Forward pass & shapes", "Universal approximation", "Parameter count", "Depth vs width experiment", "Post-test"], active=0)
    h2("من عصبون إلى شبكة في خمس خطوات", "From a neuron to a network in five steps")
    animation_player("w04_blocks", [Frame(_blocks_svg(i), caption(t), action=n, highlight=i) for i, (n, t, _) in enumerate(_BLOCKS)],
                     title_ar="قطع البناء", stages=[b[0] for b in _BLOCKS], interval_ms=2400)
    table(["الموضوع", "درس الأسبوع", "الأسس التي يعتمد عليها"],
          [("التغذية الأمامية وMLP والطبقات", "التدفق الأمامي", "الأسس 10: طبقات ومعلمات"), ("الأوزان والانحيازات والتمرير الأمامي", "التدفق الأمامي", "الأسس 9 و12: الخلية والتمرير الأمامي"),
           ("التقريب الشامل: لماذا تكفي طبقة واحدة نظريًا", "التدفق الأمامي", "الأسس 11: لماذا اللاخطية"),
           ("بناء نماذج متعددة الطبقات وعدّ المعلمات", "العمق والعرض", "الأسس 10: عدّ المعلمات؛ الوحدة 21: الطبقات والنماذج"), ("السعة وفرط التخصيص", "العمق والعرض", "الأسس 19: التعميم")],
          ["rtl", "rtl", "rtl"])
    st.markdown(
        "**في نهاية الأسبوع ستستطيع:**\n"
        "- أن تحسب يدويًا مخرج شبكة صغيرة لعميل واحد، وأن تتنبأ بأشكال كل طبقة لدفعة.\n"
        "- أن تشرح ضرب المصفوفات كـ «ملاحظة · وحدة» ولماذا نعمل بدفعات.\n"
        "- أن تشرح بصريًا لماذا تستطيع وحدات ReLU تقريب أي منحنى.\n"
        "- أن تعدّ معلمات أي MLP وتختار عمقه وعرضه بتجربة على التحقق."
    )
    practical_note("الأسس 9 و10 و12 هي المرجع؛ هذا الأسبوع يجمعها في Keras ويجري تجربة عمق/عرض حقيقية.")
    with st.container(horizontal=True):
        st.button("الأسس 10 — البنية", icon=":material/menu_book:", on_click=go, args=("foundations.architecture",), key="w04_go_arch")
        st.button("الأسس 12 — التمرير الأمامي", icon=":material/menu_book:", on_click=go, args=("foundations.forward",), key="w04_go_fwd")
        st.button("معمل بناء الشبكة", icon=":material/science:", on_click=go, args=("labs.network_builder",), key="w04_go_lab")
    intuition("كل ما في هذا الأسبوع يُختصر في سطر: `a = f(aW + b)` مكررًا. الباقي — الأشكال، المعلمات، السعة — نتائج مباشرة لهذا السطر.")
    h2("الاختبار القبلي", "Pre-test")
    quiz("course.w04.pretest", [
        Q("في شبكة أمامية، المعلومات تتدفق…", ["في الاتجاهين", "من المدخل إلى المخرج فقط", "عشوائيًا"], 1, "بلا حلقات."),
        Q("Dense(32) على مدخل بـ 10 خصائص: المعلمات", ["320", "352", "42"], 1, "10×32 + 32."),
        Q("طبقة مخفية بلا تنشيط غير خطي…", ["تزيد القدرة", "تُطوى مع التي بعدها في طبقة خطية واحدة", "تمنع التدريب"], 1, "تركيب دوال خطية = دالة خطية."),
        Q("شكل مخرج Dense(8) لدفعة من 64:", ["(8,)", "(64, 8)", "(8, 64)"], 1, "الدفعة ثابتة؛ الأعمدة = الوحدات."),
        Q("مخرج تصنيف لخمس فئات يحتاج…", ["Dense(1, sigmoid)", "Dense(5, softmax)", "Dense(5, relu)"], 1, ""),
    ], title_ar="الاختبار القبلي — الأسبوع 04")
    takeaway("الأسبوع 04 = بنية MLP بالأشكال والمعلمات والتقريب الشامل، ثم تجربة اختيار البنية بأدلة.")
    lesson_footer(LESSON, ["قطع البناء (تحريك).", "خريطة الأسبوع.", "الاختبار القبلي."])
