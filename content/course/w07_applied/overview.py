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
    id="course.w07.overview",
    title_ar="نظرة عامة على الأسبوع 07 والاختبار القبلي",
    title_en="Week 07 Overview & Pre-test",
    module="course.w07",
    order=1,
    prerequisites=["course.w06.choosing_activations", "course.w02.evaluation_baselines"],
    objectives_ar=["رؤية المشروع كاملًا في عشر محطات قبل الدخول في التفاصيل (تحريك).", "خريطة الأسبوع وروابط الأسس.", "اختبار قبلي على منهج المشروع التطبيقي."],
    terms=["baseline", "reproducibility", "regression"],
    difficulty="intermediate",
    summary_ar="مشروع كامل على أسعار العقارات: تعريف المسألة ← البيانات ← الإعداد ← خطوط الأساس ← البنية ← التدريب ← التقييم ← التشخيص ← التفسير ← تحليل الأخطاء والتقرير.",
)

_STEPS = [
    ("Problem", "**تعريف المسألة**: سعر العقار بآلاف الدنانير من المساحة والغرف والعمر والمنطقة — انحدار، ومقياسه RMSE/MAE بوحدة الهدف.", K.VIOLET),
    ("Data", "**البيانات**: 300 عقار، أربعة أعمدة خصائص وهدف؛ تقسيم 180/60/60 ببذرة ثابتة قبل أي حساب.", K.BLUE),
    ("Prep", "**الإعداد**: توحيد العددي بإحصاءات التدريب، one-hot للمنطقة، توحيد الهدف (ويُعكس قبل الإبلاغ).", K.CYAN),
    ("Baselines", "**خطوط الأساس**: المتوسط ثم الانحدار الخطي — المسطرة التي تُقاس عليها الشبكة.", K.EMERALD),
    ("Model", "**البنية**: MLP صغير 7 → 32 → 16 → 1 بلا تنشيط في المخرج، MSE، Adam.", K.AMBER),
    ("Train", "**التدريب**: إيقاف مبكر على التحقق واستعادة أفضل أوزان؛ منحنى loss/val_loss.", K.ORANGE),
    ("Evaluate", "**التقييم**: مرة واحدة على الاختبار، بالدينار، مقابل خطي الأساس.", K.PINK),
    ("Diagnose", "**التشخيص**: الفجوة، المنحنى، البواقي (هيستوغرام ومقابل التنبؤ).", K.VIOLET),
    ("Interpret", "**التفسير**: أهمية التبديل بجانب معاملات الخطي — بلا ادعاء سببي.", K.BLUE),
    ("Report", "**الأخطاء والتقرير**: أسوأ 5 وأنماطها، استقرار عبر التقسيمات، وتقرير قابل للاستنساخ.", K.EMERALD),
]


def _svg(k: int) -> str:
    s = K.svg_open(700, 150)
    for i, (n, _, col) in enumerate(_STEPS):
        row, c = divmod(i, 5)
        x, y = 12 + c * 138, 18 + row * 62
        s += K.box(x, y, 124, 44, f"{i + 1}. {n}", col, filled=i == k, size=12)
        if c < 4:
            s += K.arrow(x + 125, y + 22, x + 137, y + 22, color=col)
    s += K.arrow(12 + 4 * 138 + 62, 64, 12 + 62, 78, color=K.MUTED, dash="4 3")
    return s + "</svg>"


def render() -> None:
    lesson_header(LESSON)
    h2("ماذا سنفعل هذا الأسبوع؟", "This week")
    pipeline(["Problem", "Data prep", "Baseline", "Architecture", "Training", "Evaluation", "Diagnostics", "Interpretation", "Error analysis", "Post-test"], active=0)
    h2("المشروع في عشر محطات", "The project in ten stops")
    animation_player("w07_journey", [Frame(_svg(i), caption(t), action=n, highlight=None) for i, (n, t, _) in enumerate(_STEPS)],
                     title_ar="من السؤال إلى التقرير", interval_ms=2200)
    table(["الموضوع", "درس الأسبوع", "الأسس التي يعتمد عليها"],
          [("تعريف المسألة والبيانات وإعدادها", "بناء خط الأنابيب", "الأسس 1 و8؛ الأسبوع 02"), ("خط الأساس والبنية والتدريب", "بناء خط الأنابيب", "الأسبوعان 02 و04؛ الأسس 20 (الإيقاف المبكر)"),
           ("التقييم والتشخيص", "التشخيص والتفسير", "الأسس 18 و19؛ الأسبوع 05 (المنحنيات)"), ("التفسير وتحليل الأخطاء", "التشخيص والتفسير", "الأسبوع 01 (تنبؤ ≠ سببية)"), ("التقرير والاستنساخ", "التشخيص والتفسير", "الأسس 2 (البذرة)")],
          ["rtl", "rtl", "rtl"])
    st.markdown(
        "**في نهاية الأسبوع ستستطيع:**\n"
        "- أن تحوّل سؤالًا تطبيقيًا إلى خط أنابيب كامل بقرارات مبرَّرة.\n"
        "- أن تعدّ البيانات بلا تسريب وتبلّغ الأخطاء بوحدة الهدف.\n"
        "- أن تشخّص النموذج من المنحنيات والبواقي وتفسّره بأهمية التبديل.\n"
        "- أن تكتب تقريرًا قابلًا للاستنساخ وتتجنب ادعاء التفوق من تشغيل واحد."
    )
    practical_note("هذا الأسبوع قالب مشروع الأسبوع 14: نفّذه على بيانات المنصة هنا، ثم على بياناتك هناك.")
    with st.container(horizontal=True):
        st.button("الأسس 8 — إعداد البيانات", icon=":material/menu_book:", on_click=go, args=("foundations.prep",), key="w07_go_prep")
        st.button("الأسس 19 — التعميم", icon=":material/menu_book:", on_click=go, args=("foundations.generalization",), key="w07_go_gen")
        st.button("معمل تشخيص المنحنيات", icon=":material/science:", on_click=go, args=("labs.curves_diagnostic_lab",), key="w07_go_lab")
    intuition("المشروع الجيد ليس النموذج الأعقد بل **سلسلة القرارات المبرَّرة**: كل خطوة تجيب «لماذا هكذا؟» وكل رقم على بيانات لم تُستعمل في اتخاذ القرار.")
    h2("الاختبار القبلي", "Pre-test")
    quiz("course.w07.pretest", [
        Q("أول نموذج تبنيه في مشروع:", ["أعمق شبكة", "خط أساس بسيط", "LSTM"], 1, "المسطرة قبل النموذج."),
        Q("RMSE يُبلَّغ…", ["كنسبة", "بوحدة الهدف", "كخسارة"], 1, "بالدينار ليفهمه المستخدم."),
        Q("val_loss يرتفع من الحقبة 5 بينما loss ينخفض:", ["قصور", "فرط تخصيص → إيقاف مبكر/تنظيم", "تسريب"], 1, ""),
        Q("أهمية خاصية عالية في النموذج تعني…", ["سببية", "أن النموذج يعتمد عليها للتنبؤ", "أنها مقاسة جيدًا"], 1, ""),
        Q("إحصاءات التوحيد تُحسب على…", ["كل البيانات", "التدريب فقط", "الاختبار"], 1, "وإلا تسريب."),
    ], title_ar="الاختبار القبلي — الأسبوع 07")
    takeaway("الأسبوع 07 = المنهج كاملًا على مسألة واحدة، بقرارات مبرَّرة وأرقام على بيانات محجوزة.")
    lesson_footer(LESSON, ["المشروع في عشر محطات (تحريك).", "خريطة الأسبوع.", "الاختبار القبلي."])
