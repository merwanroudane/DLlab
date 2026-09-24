import streamlit as st

from components.animation_player import Frame, animation_player, caption
from components.callouts import intuition, practical_note, takeaway
from components.comparison import compare_table
from components.lesson_layout import h2, lesson_footer, lesson_header
from components.quiz import Q, quiz
from components import svgkit as K
from core.models import Lesson
from core.routing import go
from core.rtl import pipeline, table
from labs.datasets import loan_default

LESSON = Lesson(
    id="course.w02.overview",
    title_ar="نظرة عامة على الأسبوع 02 والاختبار القبلي",
    title_en="Week 02 Overview & Pre-test",
    module="course.w02",
    order=1,
    prerequisites=["course.w01.applications", "foundations.ml.task_types", "foundations.eval.splits_metrics"],
    objectives_ar=["خريطة الأسبوع وربط كل موضوع بدرسه التأسيسي.", "التعرف على المثال الذي يرافقنا طوال الأسبوع (تعثر القروض).", "اختبار قبلي يوجّهك إلى ما تراجعه."],
    terms=["machine_learning", "model", "target", "baseline", "validation_set"],
    difficulty="beginner",
    summary_ar="أنواع النماذج ← الانحدار/التصنيف ← التقييم والتقسيم والمقاييس وخطوط الأساس ← التعميم ← من ML إلى DL.",
)

_JOURNEY = [
    ("Question", "**السؤال**: من سيتعثر؟ هدف فئوي ⇒ **تصنيف ثنائي** (درس أنواع النماذج).", K.VIOLET),
    ("Model", "**أبسط نموذج**: انحدار لوجستي = عصبون واحد برأس sigmoid. نشاهده يتدرب ونقرأ معاملاته كنسب أرجحية.", K.BLUE),
    ("Split", "**التقسيم**: 60/20/20 طبقي؛ التحقق للقرارات، والاختبار مرة واحدة (درس التقييم).", K.CYAN),
    ("Baseline", "**خط الأساس**: الفئة الغالبة وقاعدة خبير — المسطرة التي يُقاس عليها كل نموذج.", K.EMERALD),
    ("Metric", "**المقياس والعتبة**: مصفوفة الالتباس، الصحة/الاستدعاء، ROC، وعتبة تُختار بكلفة الخطأ.", K.AMBER),
    ("Generalize", "**التعميم**: شجرة تتعمق حتى تحفظ الضجيج — فجوة التدريب/التحقق.", K.ORANGE),
    ("ML → DL", "**الشبكة**: متى تتفوق MLP على اللوجستي، وكيف تصنع الطبقة المخفية الانحناء.", K.PINK),
]


def _journey_svg(k: int) -> str:
    s = K.svg_open(680, 120)
    for i, (name, _, col) in enumerate(_JOURNEY):
        x = 8 + i * 96
        s += K.box(x, 35, 84, 46, name, col, filled=i == k, size=11)
        if i < len(_JOURNEY) - 1:
            s += K.arrow(x + 85, 58, x + 95, 58, color=col)
    s += K.text(340, 108, "one running example: loan default (400 customers)", size=11, color=K.MUTED)
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["Model types", "Regression / Classification", "Split · Metrics · Baselines", "Generalization", "ML → DL", "Post-test"], active=0)
    table(["الموضوع", "درس الأسبوع", "الأسس التي يعتمد عليها"],
          [("أنواع النماذج والمهام", "أنواع النماذج", "الأسس 7: النماذج والمهام؛ الأسس 1: الهدف"), ("الانحدار والتصنيف بمثال", "أنواع النماذج", "الأسس 7: الانحدار الخطي واللوجستي"),
           ("تدريب/تحقق/اختبار، المقاييس، خطوط الأساس", "التقييم وخطوط الأساس", "الأسس 8: التقسيم والتسريب؛ الأسس 18: المقاييس"), ("التعميم", "التقييم وخطوط الأساس", "الأسس 19"),
           ("من التعلم الآلي إلى العميق", "من ML إلى DL", "الأسس 9–10: من الانحدار إلى الخلية والشبكة")], ["rtl", "rtl", "rtl"])
    h2("رحلة الأسبوع على مثال واحد", "The week's journey on one example")
    animation_player("w02_journey", [Frame(_journey_svg(i), caption(c), action=n) for i, (n, c, _) in enumerate(_JOURNEY)],
                     title_ar="سبع محطات من السؤال إلى الشبكة", interval_ms=2200)
    st.markdown("**بيانات الأسبوع**: 400 عميل (بيانات تعليمية مولَّدة بآلية معروفة، مع قيم مفقودة عمدًا). أول الصفوف:")
    st.dataframe(loan_default(n=400, seed=7).head(8), hide_index=True)
    compare_table(["العمود", "الدور", "النوع"],
                  [("income, age, num_late_payments, debt_ratio", "خصائص عددية", "عددي"), ("city, employment", "خصائص فئوية", "فئوي اسمي"),
                   ("defaulted", "**الهدف**", "ثنائي 0/1"), ("customer_id", "معرّف — **ليس** خاصية", "لا يدخل النموذج أبدًا")],
                  ["ltr", "rtl", "rtl"])
    st.markdown(
        "**في نهاية الأسبوع ستستطيع:**\n"
        "- أن تحدد نوع أي مسألة ومخرجها وخسارتها ومقياسها.\n"
        "- أن تقرأ معاملات انحدار لوجستي كنسب أرجحية.\n"
        "- أن تقسم البيانات دون تسريب وتبني خط أساس وتختار عتبة بحسب كلفة الخطأ.\n"
        "- أن تشخّص القصور وفرط التخصيص من فجوة التدريب/التحقق.\n"
        "- أن تشرح بصريًا كيف تصنع طبقة مخفية حدًّا منحنيًا، ومتى لا تستحق الشبكة كلفتها."
    )
    practical_note("هذا الأسبوع لا يعيد الأسس بل **يُشغّلها معًا** على مسألة واحدة (تعثر القروض) من السؤال إلى الحكم. إن كانت الوحدات 7 و8 و18 و19 غير مألوفة فراجعها أولًا.")
    with st.container(horizontal=True):
        st.button("الأسس 7 — التعلم الآلي", icon=":material/menu_book:", on_click=go, args=("foundations.ml",), key="w02_go_ml")
        st.button("الأسس 8 — إعداد البيانات", icon=":material/menu_book:", on_click=go, args=("foundations.prep",), key="w02_go_prep")
        st.button("الأسس 18 — التقييم", icon=":material/menu_book:", on_click=go, args=("foundations.eval",), key="w02_go_eval")
    intuition("التعلم العميق تعلم آلي بنماذج أعمق: كل ما في هذا الأسبوع (التقسيم، خط الأساس، المقاييس، التعميم) يبقى كما هو حين تصبح النماذج شبكات بملايين المعلمات.")
    h2("الاختبار القبلي", "Pre-test")
    quiz("w02.pretest", [
        Q("التنبؤ بسعر عقار بالدينار هو…", ["تصنيف", "انحدار", "تجميع"], 1, "هدف عددي متصل."),
        Q("مجموعة التحقق تُستخدم لـ…", ["تدريب المعلمات", "اختيار المعلمات الفائقة والإيقاف", "التقرير النهائي"], 1, "الاختبار للتقرير مرة واحدة."),
        Q("فئة إيجابية 3% فقط: الدقة 97% تعني…", ["نموذج ممتاز", "ربما لا شيء: التنبؤ بالفئة الغالبة يعطيها", "تسريب"], 1, "خط الأساس."),
        Q("فجوة التعميم هي الفرق بين…", ["التدريب والتحقق", "الانحدار والتصنيف", "الدقة والخسارة"], 0, "أداء التدريب مقابل المحجوز."),
        Q("هل يدخل `customer_id` كخاصية في النموذج؟", ["نعم", "لا؛ معرّف بلا معنى تنبؤي وقد يسبب حفظًا", "فقط بعد التحجيم"], 1, "المعرّفات ليست خصائص."),
        Q("متى تتفوق الشبكة العصبية على الانحدار اللوجستي عادةً؟", ["دائمًا", "عندما تكون العلاقة غير خطية والبيانات كافية", "أبدًا"], 1, "غير خطية + بيانات."),
    ], title_ar="الاختبار القبلي — الأسبوع 02")
    takeaway("الأسبوع 02 = المنهج العلمي للتعلم الآلي على مسألة واحدة: نوع المهمة، تقسيم، خط أساس، مقياس، تعميم — ثم لماذا الشبكة.")
    lesson_footer(LESSON, ["خريطة الأسبوع ورحلته.", "بيانات القروض وأدوار الأعمدة.", "الاختبار القبلي."])
